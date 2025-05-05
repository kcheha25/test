import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --------- Charger modèle DINO pré-entraîné ---------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger DINOv2 ViT-Small
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
model.to(device)

# --------- Prétraitement de l'image ---------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),   # Taille d'entrée pour ViT
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Charger l'image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = image.shape[:2]

input_tensor = transform(image).unsqueeze(0).to(device)

# --------- Extraire les features avec DINO ---------
with torch.no_grad():
    features = model.forward_features(input_tensor)['x_norm_patchtokens']  # (B, N_patches, D)
    features = features[0]  # (N_patches, D)

# --------- Clustering KMeans sur les features ---------
n_clusters = 5  # Choisir nombre de classes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features.cpu().numpy())

# --------- Remettre les labels sous forme d'image ---------
h, w = 14, 14  # ViT-Small sort 14x14 patches
segmentation = labels.reshape(h, w)
segmentation = cv2.resize(segmentation.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

# --------- Affichage ---------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='tab20')  # Colormap colorée
plt.title('Segmentation DINO + KMeans')
plt.axis('off')

plt.show()


import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# --------- Charger modèle DINO pré-entraîné ---------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger DINOv2 ViT-Small
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
model.to(device)

# --------- Prétraitement de l'image ---------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Charger l'image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = image.shape[:2]

input_tensor = transform(image).unsqueeze(0).to(device)

# --------- Extraire les features avec DINO ---------
with torch.no_grad():
    features = model.forward_features(input_tensor)['x_norm_patchtokens']  # (B, N_patches, D)
    features = features[0]  # (N_patches, D)

# --------- Reformater les features pour correspondre à l'image ---------
h_feat, w_feat = 14, 14  # Taille patchs ViT
features_map = features.reshape(h_feat, w_feat, -1).cpu().numpy()
features_map = cv2.resize(features_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

# --------- Générer Superpixels avec SLIC ---------
n_segments = 200  # Choisis nombre de superpixels
segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)

# --------- Moyenne des features dans chaque superpixel ---------
output_labels = np.zeros(segments.shape, dtype=np.int32)

superpixel_features = []
for seg_val in np.unique(segments):
    mask = segments == seg_val
    segment_features = features_map[mask]
    mean_feature = segment_features.mean(axis=0)
    superpixel_features.append(mean_feature)

superpixel_features = np.array(superpixel_features)

# --------- Clustering des superpixels (optionnel) ---------
from sklearn.cluster import KMeans

n_clusters = 5  # Choisis nombre de classes finales
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(superpixel_features)

# Assigner cluster à chaque superpixel
for seg_val, cluster_id in zip(np.unique(segments), cluster_labels):
    output_labels[segments == seg_val] = cluster_id

# --------- Affichage ---------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(label2rgb(segments, image, kind='avg'))
plt.title('Superpixels SLIC')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_labels, cmap='tab20')
plt.title('Segmentation finale (DINO + Superpixels)')
plt.axis('off')

plt.tight_layout()
plt.show()


import torch
import torchvision.transforms as T
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# --------- Charger modèle DINO pré-entraîné ---------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger DINOv2 ViT-Small
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
model.to(device)

# --------- Prétraitement de l'image ---------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),  # Taille d'entrée pour ViT
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Charger l'image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = image.shape[:2]

input_tensor = transform(image).unsqueeze(0).to(device)

# --------- Extraire les features avec DINO ---------
with torch.no_grad():
    features = model.forward_features(input_tensor)['x_norm_patchtokens']  # (B, N_patches, D)
    features = features[0]  # (N_patches, D)

# --------- Reformater les features pour correspondre à l'image ---------
h_feat, w_feat = 14, 14  # Taille patchs ViT
features_map = features.reshape(h_feat, w_feat, -1).cpu().numpy()
features_map = cv2.resize(features_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

# --------- Clustering Hiérarchique Agglomératif (CAH) ---------
# Nous transformons les features en 2D et appliquons CAH pour une segmentation douce
X = features_map.reshape(-1, features_map.shape[-1])  # (n_patches, D)

# Appliquer CAH (on utilise l'algorithme AgglomerativeClustering)
n_clusters = 5  # Choisir le nombre de clusters final (peut aussi être déterminé dynamiquement)
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
labels = clustering.fit_predict(X)

# --------- Remettre les labels sous forme d'image ---------
segmentation = labels.reshape(h_feat, w_feat)
segmentation = cv2.resize(segmentation.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

# --------- Affichage ---------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(features_map)
plt.title('Features DINO')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(segmentation, cmap='tab20')  # Affichage des segments
plt.title('Segmentation avec DINO + CAH')
plt.axis('off')

plt.tight_layout()
plt.show()


import torch
import torchvision.transforms as T
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --------- Charger modèle DINO pré-entraîné ---------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger DINOv2 ViT-Small
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
model.to(device)

# --------- Prétraitement de l'image ---------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),  # Taille d'entrée pour ViT
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Charger l'image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = image.shape[:2]

input_tensor = transform(image).unsqueeze(0).to(device)

# --------- Extraire les features avec DINO ---------
with torch.no_grad():
    features = model.forward_features(input_tensor)['x_norm_patchtokens']  # (B, N_patches, D)
    features = features[0]  # (N_patches, D)

# --------- Reformater les features pour correspondre à l'image ---------
h_feat, w_feat = 14, 14  # Taille patchs ViT
features_map = features.reshape(h_feat, w_feat, -1).cpu().numpy()
features_map = cv2.resize(features_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

# --------- Clustering Hiérarchique Agglomératif (CAH) ---------
# Nous transformons les features en 2D
X = features_map.reshape(-1, features_map.shape[-1])  # (n_patches, D)

# Méthode du coude et indice de silhouette pour déterminer le nombre optimal de clusters
inertia = []
silhouette_scores = []
max_clusters = 10  # Nombre maximal de clusters à tester

for n_clusters in range(2, max_clusters + 1):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(X)

    # Calculer l'inertie (somme des distances intra-clusters)
    inertia.append(clustering.inertia_ if hasattr(clustering, 'inertia_') else None)

    # Calculer l'indice de silhouette
    if len(set(labels)) > 1:  # Vérifier que le clustering a plus de 1 cluster
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    else:
        silhouette_scores.append(-1)

# Affichage des résultats de la méthode du coude et de l'indice de silhouette
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), inertia, marker='o')
plt.title('Méthode du Coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.xticks(range(2, max_clusters + 1))

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Indice de Silhouette')
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Silhouette')
plt.xticks(range(2, max_clusters + 1))

plt.tight_layout()
plt.show()

# --------- Choisir le meilleur nombre de clusters basé sur l'indice de silhouette
best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'Nombre optimal de clusters selon l\'indice de silhouette: {best_n_clusters}')

#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import io

# Charger l'image
image = io.imread('image.jpg')

# Reshaper l'image pour l'appliquer à DBSCAN
pixels = image.reshape((-1, 3))

# Appliquer DBSCAN pour segmenter l'image
dbscan = DBSCAN(eps=10, min_samples=10)
labels = dbscan.fit_predict(pixels)

# Remodeler les résultats en forme de l'image
segmented_image = labels.reshape(image.shape[0], image.shape[1])

# Afficher l'image segmentée
plt.imshow(segmented_image)
plt.show()

from matplotlib.colors import ListedColormap

colors = plt.cm.get_cmap('tab20', n_clusters)
custom_cmap = ListedColormap(colors(np.arange(n_clusters)))

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.cluster import spectral_clustering
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt

# Charger le modèle Stable Diffusion
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Fonction pour extraire les self-attentions
def get_self_attentions(image_path):
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to("cuda")

    # Inversion DDIM (10 steps comme dans l'article)
    latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215
    inverted_latents = pipe.scheduler.invert(latents, num_inference_steps=10)

    # Extraire les self-attentions du décodeur
    with torch.no_grad():
        output = pipe.unet(inverted_latents, timestep=pipe.scheduler.timesteps[0], return_dict=True)
    
    # Agréger les self-attentions sur toutes les couches/résolutions
    attentions = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and "processor" in name:  # Couches d'auto-attention
            attn = module.attention_probs  # Shape: (batch, heads, seq_len, seq_len)
            attentions.append(attn.mean(dim=1))  # Moyenne sur les têtes

    # Agréger toutes les attentions (simplifié par rapport à l'article)
    aggregated_attention = torch.stack(attentions).mean(dim=0).squeeze()  # Shape: (seq_len, seq_len)
    return aggregated_attention.cpu().numpy()

# Fonction pour appliquer Normalized Cuts
def normalized_cuts(adjacency_matrix, n_clusters=5):
    # Calculer le Laplacien
    laplacian_matrix = laplacian(adjacency_matrix, normed=True)
    
    # Clustering spectral
    labels = spectral_clustering(
        laplacian_matrix,
        n_clusters=n_clusters,
        eigen_solver="amg",
        random_state=0
    )
    return labels

# Fonction pour visualiser la segmentation
def visualize_segmentation(image_path, labels):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.imshow(labels.reshape(64, 64), alpha=0.5, cmap="jet")
    plt.colorbar()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "chemin/vers/ton/image.jpg"  # Remplace par ton image
    
    # 1. Extraire les self-attentions
    attention_matrix = get_self_attentions(image_path)
    
    # 2. Construire la matrice d'adjacence (dot product comme dans l'article)
    adjacency_matrix = attention_matrix @ attention_matrix.T
    
    # 3. Appliquer Normalized Cuts
    n_clusters = 5  # Nombre de segments (peut être ajusté)
    labels = normalized_cuts(adjacency_matrix, n_clusters)
    
    # 4. Visualiser
    visualize_segmentation(image_path, labels)