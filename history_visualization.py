import dill
import matplotlib.pyplot as plt
import os

save_dir = "/data/MMM.u2/mcwon/src/plot"
os.makedirs(save_dir, exist_ok=True)

with open("/data/MMM.u2/mcwon/saved/MMM2____mmm/history____mmm.pkl", "rb") as f:
    history = dill.load(f)

epochs = range(1, len(history["adm_ja_CID"]) + 1)

ja_cid = history["adm_ja_CID"]
ja_atc = history["adm_ja_ATC3"]

f1_cid = history["adm_f1_CID"]
f1_atc = history["adm_f1_ATC3"]

prauc_cid = history["adm_prauc_CID"]
prauc_atc = history["adm_prauc_ATC3"]

auroc_cid = history["adm_auroc_CID"]
auroc_atc = history["adm_auroc_ATC3"]

ddi = history["ddi_rate"]
loss = history["loss"]
val_loss = history.get("val_loss", None)


min_ddi_idx = ddi.index(min(ddi))
min_loss_idx = loss.index(min(loss))
if val_loss:
    min_val_loss_idx = val_loss.index(min(val_loss))


plt.figure(figsize=(24, 10))

plt.subplot(2, 3, 3)
plt.plot(epochs, auroc_cid, marker='o', label="CID")
plt.plot(epochs, prauc_atc, marker='x', label="ATC3")
plt.title("AUROC over Epochs")
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(epochs, ja_cid, marker='o', label="CID")
plt.plot(epochs, ja_atc, marker='x', label="ATC3")
plt.title("Jaccard over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Jaccard")
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs, f1_cid, marker='o', label="CID")
plt.plot(epochs, f1_atc, marker='x', label="ATC3")
plt.title("F1 Score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs, prauc_cid, marker='o', label="CID")
plt.plot(epochs, prauc_atc, marker='x', label="ATC3")
plt.title("PRAUC over Epochs")
plt.xlabel("Epoch")
plt.ylabel("PRAUC")
plt.legend()

plt.subplot(2, 3, 1)
min_ddi_idx = ddi.index(min(ddi))
plt.plot(epochs, ddi, marker='o')
plt.plot(epochs[min_ddi_idx], ddi[min_ddi_idx], 'ro')
plt.title("DDI Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("DDI Rate")

plt.subplot(2, 3, 2)
min_loss_idx = loss.index(min(loss))
plt.plot(epochs, loss, marker='o', label="Train Loss")
if val_loss:
    min_val_loss_idx = val_loss.index(min(val_loss))
    plt.plot(epochs, val_loss, marker='o', label="Validation Loss", color='orange')
    plt.plot(epochs[min_val_loss_idx], val_loss[min_val_loss_idx], 'ro')
plt.plot(epochs[min_loss_idx], loss[min_loss_idx], 'go')
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plot_path = os.path.join(save_dir, "training_metrics_over_epochs_cid_atc3.png")
plt.savefig(plot_path)
print(f"Plot has been saved: {plot_path}")

plt.show()