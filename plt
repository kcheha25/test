import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Charger les données JSON
file_path = "chromatogrammes.json"
df = pd.read_json(file_path)

# Filtrer les chromatogrammes avec pics
df = df.dropna(subset=['pics'])

# Troncature des chromatogrammes
def truncate(row):
    mask = np.array(row["x"]) <= 150
    row["x"] = np.array(row["x"])[mask].tolist()
    row["y"] = np.array(row["y"])[mask].tolist()
    return row

df = df.apply(truncate, axis=1)

sequence_length = 1000  # Taille des segments

# Créer les dossiers de sortie
os.makedirs("output/images", exist_ok=True)
os.makedirs("output/labels", exist_ok=True)

def save_segment_and_labels(x, y, pics, chromatogram_index, segment_index, segment_size=1000):
    # Normalisation des intensités
    max_intensity = np.max(y)
    if max_intensity > 0:
        y = y / max_intensity  # Normaliser l'intensité

    # Segmenter les données
    start_idx = segment_index * segment_size
    end_idx = start_idx + segment_size

    segment_x = np.array(x[start_idx:end_idx]) / 150.0  # Normalisation de x
    segment_y = np.array(y[start_idx:end_idx])

    x_min, x_max = segment_x.min(), segment_x.max()

    # Création de l'image
    plt.figure(figsize=(6.4, 6.4), dpi=100)  # Image 640x640 pixels
    plt.plot(segment_x, segment_y, color="black", linewidth=2)
    plt.axis("off")

    # Sauvegarde de l'image
    image_filename = f"chromatogramme_{chromatogram_index}_segment{segment_index}.png"
    image_path = os.path.join("output/images", image_filename)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Sauvegarde des annotations YOLO
    label_filename = f"chromatogramme_{chromatogram_index}_segment{segment_index}.txt"
    label_path = os.path.join("output/labels", label_filename)

    with open(label_path, "w") as f:
        for pic_time, data in pics.items():
            pic_time = float(pic_time) / 150.0
            borne_avant_time, borne_apres_time = data[1] / 150.0, data[2] / 150.0

            if start_idx <= np.argmin(np.abs(np.array(x) - pic_time * 150)) < end_idx:
                # Indices correspondant aux pics
                pic_idx = np.argmin(np.abs(segment_x - pic_time))
                borne_avant_idx = np.argmin(np.abs(segment_x - borne_avant_time))
                borne_apres_idx = np.argmin(np.abs(segment_x - borne_apres_time))

                x_center = (pic_time - x_min) / (x_max - x_min)
                width = (borne_apres_time - borne_avant_time) / (x_max - x_min)
                y_center = segment_y[pic_idx]
                height = y_center  # Hauteur = intensité du pic

                # YOLO format: class x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Itérer sur les chromatogrammes et segmenter
for chromatogram_index, row in df.iterrows():
    x = np.array(row["x"])
    y = np.array(row["y"])
    pics = row["pics"]

    num_segments = len(x) // sequence_length

    for segment_index in range(num_segments):
        save_segment_and_labels(x, y, pics, chromatogram_index, segment_index, sequence_length)

print("Segmentation terminée ! Images et annotations sauvegardées.")
