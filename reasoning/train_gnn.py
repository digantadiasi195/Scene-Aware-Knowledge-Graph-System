# reasoning/train_gnn.py

import os
import random
import numpy as np

# Fix OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from tqdm import tqdm

from gnn_reasoner import GATReasoner


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "../graph/graph_dataset.pt"

MODEL_SAVE_PATH = "best_gnn_model.pth"

NUM_CLASSES = 5
NUM_NODE_FEATURES = 60

HIDDEN_DIM = 64
GAT_HEADS = 4

EPOCHS = 100
BATCH_SIZE = 64

LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        f"✅ Using device: {device} "
        f"({torch.cuda.get_device_name(0)})"
    )

else:

    device = torch.device("cpu")

    print(
        "⚠️ CUDA unavailable. Using CPU."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_and_prepare_data():

    print(
        f"Loading graph dataset from {DATASET_PATH}..."
    )

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            DATASET_PATH
        )


    dataset = torch.load(
        DATASET_PATH,
        weights_only=False
    )


    print(
        "Total graph samples:",
        len(dataset)
    )


    # ----------------------------
    # Class weights
    # ----------------------------

    labels = [
        data.y.item()
        for data in dataset
    ]


    unique_classes = np.unique(labels)


    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )


    full_weights = torch.ones(
        NUM_CLASSES,
        dtype=torch.float32
    )


    for cls, w in zip(
        unique_classes,
        weights
    ):
        full_weights[cls] = w


    class_weights = full_weights.to(device)


    print(
        "Calculated Full Class Weights:",
        class_weights
    )


    # ----------------------------
    # Shuffle
    # ----------------------------

    random.shuffle(dataset)


    split_idx = int(
        len(dataset) * TRAIN_SPLIT
    )


    train_dataset = dataset[:split_idx]

    val_dataset = dataset[split_idx:]


    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # Distribution check

    from collections import Counter


    print(
        "\nTrain distribution:"
    )

    print(
        Counter(
            [
                g.y.item()
                for g in train_dataset
            ]
        )
    )


    print(
        "\nValidation distribution:"
    )

    print(
        Counter(
            [
                g.y.item()
                for g in val_dataset
            ]
        )
    )


    return (
        train_dataset,
        val_dataset,
        class_weights
    )



# ============================================================
# TRAINING
# ============================================================

def train():


    (
        train_dataset,
        val_dataset,
        class_weights
    ) = load_and_prepare_data()



    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    # ----------------------------
    # Model
    # ----------------------------

    model = GATReasoner(

        num_node_features=NUM_NODE_FEATURES,

        hidden_dim=HIDDEN_DIM,

        num_classes=NUM_CLASSES,

        gat_heads=GAT_HEADS

    ).to(device)



    print(
        "\nModel initialized"
    )



    optimizer = Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=5e-4

    )



    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=5

    )



    criterion = torch.nn.CrossEntropyLoss(

        weight=class_weights

    )



    best_f1 = 0.0



    print(
        "\nStarting training..."
    )


    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):


        # -------------------------
        # Training
        # -------------------------

        model.train()


        total_loss = 0

        correct = 0

        total = 0



        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        ):


            batch = batch.to(device)


            optimizer.zero_grad()



            output = model(

                batch.x,

                batch.edge_index,

                batch.batch

            )



            loss = criterion(

                output,

                batch.y

            )



            loss.backward()

            optimizer.step()



            total_loss += (
                loss.item()
                *
                batch.num_graphs
            )


            prediction = output.argmax(
                dim=1
            )


            correct += int(
                (prediction == batch.y)
                .sum()
            )


            total += batch.num_graphs



        train_loss = (
            total_loss / total
        )


        train_acc = (
            correct / total
        )



        # -------------------------
        # Validation
        # -------------------------

        model.eval()


        val_correct = 0

        val_total = 0


        all_preds = []

        all_labels = []



        with torch.no_grad():


            for batch in val_loader:


                batch = batch.to(device)



                output = model(

                    batch.x,

                    batch.edge_index,

                    batch.batch

                )


                prediction = output.argmax(
                    dim=1
                )



                val_correct += int(
                    (prediction == batch.y)
                    .sum()
                )


                val_total += batch.num_graphs



                all_preds.extend(
                    prediction.cpu().numpy()
                )


                all_labels.extend(
                    batch.y.cpu().numpy()
                )



        val_acc = (
            val_correct / val_total
        )



        macro_f1 = f1_score(

            all_labels,

            all_preds,

            labels=[0,1,3,4,8],

            average="macro",
            zero_division = 0

        )



        scheduler.step(
            macro_f1
        )



        print(

            f"Epoch {epoch:03d} | "

            f"Loss {train_loss:.4f} | "

            f"Train Acc {train_acc:.4f} | "

            f"Val Acc {val_acc:.4f} | "

            f"Macro F1 {macro_f1:.4f}"

        )



        # Save best F1 model

        if macro_f1 > best_f1:


            best_f1 = macro_f1


            torch.save(

                model.state_dict(),

                MODEL_SAVE_PATH

            )


            print(

                f"  💾 Saved best model "
                f"(F1={best_f1:.4f})"

            )



    print(
        "\nTraining completed"
    )


    print(
        "Best Macro F1:",
        best_f1
    )



    # ========================================================
    # Final evaluation report
    # ========================================================


    print(
        "\nLoading best model..."
    )


    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH
        )
    )


    model.eval()


    preds = []

    labels = []



    with torch.no_grad():

        for batch in val_loader:


            batch = batch.to(device)


            output = model(

                batch.x,

                batch.edge_index,

                batch.batch

            )


            pred = output.argmax(
                dim=1
            )


            preds.extend(
                pred.cpu().numpy()
            )


            labels.extend(
                batch.y.cpu().numpy()
            )



    print(
        "\nClassification Report"
    )


    print(
        classification_report(

            labels,

            preds,

            labels=[0,1,3,4,8],

            target_names=[
                "a1",
                "a2",
                "a4",
                "a5",
                "a9"
            ]

        )
    )


    print(
        "\nConfusion Matrix"
    )


    print(
        confusion_matrix(
            labels,
            preds,
            labels=[0,1,3,4,8]
        )
    )



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()