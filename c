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

n_clusters = 3
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', random_state=42)
labels = spectral.fit_predict(X).reshape(gray.shape)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from scipy.ndimage import sobel
from skimage.segmentation import find_boundaries
import networkx as nx

# --------- Étape 1 : Chargement et prétraitement ---------
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------- Étape 2 : Clustering KMeans ---------
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
labels = kmeans.labels_.reshape(gray.shape)

# --------- Étape 3 : Calcul du gradient pour la pondération ---------
gradient = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))

# --------- Étape 4 : Construction du graphe ---------
G = nx.Graph()

# Initialiser chaque cluster comme nœud
for cluster_id in range(n_clusters):
    G.add_node(cluster_id)

# Trouver les frontières
boundaries = find_boundaries(labels, mode='outer')

# Analyser les paires de labels voisins
for y in range(1, labels.shape[0]-1):
    for x in range(1, labels.shape[1]-1):
        if boundaries[y, x]:
            label = labels[y, x]
            neighbors = np.unique(labels[y-1:y+2, x-1:x+2])
            for neighbor in neighbors:
                if neighbor != label:
                    edge = tuple(sorted((label, neighbor)))
                    weight = gradient[y, x]
                    if G.has_edge(*edge):
                        G[edge[0]][edge[1]]['weights'].append(weight)
                    else:
                        G.add_edge(edge[0], edge[1], weights=[weight])

# Moyenne des poids (gradient) pour chaque arête
for u, v, data in G.edges(data=True):
    data['weight'] = np.mean(data['weights'])

# --------- Étape 5 : Fusion via Graph-Cut (simple seuil) ---------
threshold = 15  # à ajuster selon le contraste
to_merge = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]

# Fusion des clusters avec Union-Find
parent = list(range(n_clusters))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

for u, v in to_merge:
    union(u, v)

# Regrouper les clusters fusionnés
new_labels = np.zeros_like(labels)
label_map = {}
new_id = 0
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        root = find(labels[y, x])
        if root not in label_map:
            label_map[root] = new_id
            new_id += 1
        new_labels[y, x] = label_map[root]

# --------- Étape 6 : Affichage ---------
colors = plt.cm.get_cmap('tab20', new_id)
custom_cmap = ListedColormap(colors(np.arange(new_id)))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Image Grayscale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(labels, cmap=ListedColormap(plt.cm.get_cmap('tab20', n_clusters)(np.arange(n_clusters))))
plt.title('KMeans Initial')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(new_labels, cmap=custom_cmap)
plt.title('Clusters Fusionnés (Graph-Cut)')
plt.axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_objects, closing, square
from skimage.measure import label, regionprops

# Chargement et conversion en niveaux de gris
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Clustering KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
labels = kmeans.labels_.reshape(gray.shape)

# Traitement morphologique et filtrage
processed_labels = np.zeros_like(labels)

for cluster_id in range(n_clusters):
    mask = (labels == cluster_id).astype(np.uint8)
    mask_closed = closing(mask, square(5))
    mask_cleaned = remove_small_objects(mask_closed.astype(bool), min_size=500)
    processed_labels[mask_cleaned] = cluster_id + 1  # éviter 0 pour les régions

# Étiquetage des régions
labeled = label(processed_labels)

# Extraire les centroïdes des régions
regions = regionprops(labeled)
centroids = np.array([r.centroid for r in regions])
labels_map = np.array([r.label for r in regions])

# Calcul des distances entre toutes les paires de régions
dist_matrix = cdist(centroids, centroids)
np.fill_diagonal(dist_matrix, np.inf)  # éviter la fusion avec soi-même

# Seuil de fusion (en pixels)
distance_threshold = 50

# Initialiser un dictionnaire de fusions
merged_groups = []
visited = set()

for i in range(len(centroids)):
    if i in visited:
        continue
    group = {i}
    for j in range(len(centroids)):
        if dist_matrix[i, j] < distance_threshold:
            group.add(j)
    visited.update(group)
    merged_groups.append(group)

# Nouvelle image avec zones fusionnées
final_labels = np.zeros_like(labeled)

for new_label, group in enumerate(merged_groups, 1):
    for idx in group:
        final_labels[labeled == labels_map[idx]] = new_label

# Affichage
colors = plt.cm.get_cmap('tab20', len(merged_groups))
custom_cmap = ListedColormap(colors(np.arange(len(merged_groups))))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Image Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_labels, cmap=custom_cmap)
plt.title('Zones fusionnées (proximité < seuil)')
plt.axis('off')
plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Charger l'image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# --------- KMeans (3 classes) ---------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
labels_kmeans = kmeans.labels_.reshape(h, w)

# --------- CRF sur KMeans ---------
def apply_crf(image_rgb, labels, n_classes, gt_prob=0.7, iter=5):
    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], n_classes)
    unary = unary_from_labels(labels.astype(np.int32), n_classes, gt_prob=gt_prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_rgb, compat=10)
    Q = d.inference(iter)
    refined = np.argmax(Q, axis=0).reshape(labels.shape)
    return refined

labels_crf = apply_crf(image_rgb, labels_kmeans, n_classes=3)

# --------- Fusion des classes ---------
labels_fused = labels_crf.copy()
labels_fused[labels_crf == 2] = 0  # fusionne classe 2 avec 0
labels_fused[labels_crf == 1] = 1  # remappe classe 1 en 1

# --------- Préparation pour Watershed sur la classe 1 ---------
binary_mask = (labels_fused == 1).astype(np.uint8)

# Remplissage de trous uniquement (pas de fermeture)
binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)

# Calcul de la distance au fond
distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

# Détection des maxima locaux comme marqueurs
local_maxi = peak_local_max(distance, indices=False, labels=binary_mask)
markers = ndi.label(local_maxi)[0]

# Appliquer Watershed
labels_ws = cv2.watershed(image_rgb, markers.astype(np.int32))

# Les objets sont marqués par des entiers > 1, on les affiche
watershed_mask = (labels_ws > 1).astype(np.uint8)

# --------- Affichage ---------
plt.figure(figsize=(18, 10))

plt.subplot(1, 3, 1)
plt.imshow(labels_fused, cmap=ListedColormap(plt.cm.tab10.colors[:2]))
plt.title('Labels fusionnés (0/1)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(distance, cmap='jet')
plt.title('Distance Transform')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(watershed_mask, cmap='gray')
plt.title('Segmentation par Watershed')
plt.axis('off')

plt.tight_layout()
plt.show()

from scipy.ndimage import binary_fill_holes, convolve
import numpy as np

# Masques binaires initiaux (à partir de labels_fused)
mask_class_0 = (labels_fused == 0).astype(np.uint8)
mask_class_1 = (labels_fused == 1).astype(np.uint8)

# Remplissage des trous (indépendamment)
filled_class_0 = binary_fill_holes(mask_class_0).astype(np.uint8)
filled_class_1 = binary_fill_holes(mask_class_1).astype(np.uint8)

# Pixels où les résultats diffèrent (i.e. un est 1, l'autre est 0)
diff_mask = filled_class_0 != filled_class_1

# Compter les classes dans le voisinage (3x3)
kernel = np.ones((3, 3), dtype=np.uint8)
neigh_class_0 = convolve((labels_fused == 0).astype(np.uint8), kernel, mode='constant')
neigh_class_1 = convolve((labels_fused == 1).astype(np.uint8), kernel, mode='constant')

# Créer une carte vide
labels_fused_clean = np.full_like(labels_fused, fill_value=255)  # 255 = indéfini

# Appliquer directement les pixels où les deux classes sont d'accord
labels_fused_clean[filled_class_0 & filled_class_1 == 1] = 1  # les deux sont 1 → classe 1
labels_fused_clean[filled_class_0 & filled_class_1 == 0] = 0  # les deux sont 0 → classe 0

# Pour les pixels conflictuels (différence entre les deux)
dominant_class_0 = (neigh_class_0 > neigh_class_1) & diff_mask
dominant_class_1 = (neigh_class_1 >= neigh_class_0) & diff_mask  # égalité → classe 1

labels_fused_clean[dominant_class_0] = 0
labels_fused_clean[dominant_class_1] = 1
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

# Charger l'image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# --------- KMeans (3 classes) ---------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
labels_kmeans = kmeans.labels_.reshape(h, w)

# --------- CRF sur KMeans ---------
def apply_crf(image_rgb, labels, n_classes, gt_prob=0.7, iter=5):
    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], n_classes)
    unary = unary_from_labels(labels.astype(np.int32), n_classes, gt_prob=gt_prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_rgb, compat=10)
    Q = d.inference(iter)
    refined = np.argmax(Q, axis=0).reshape(labels.shape)
    return refined

labels_crf = apply_crf(image_rgb, labels_kmeans, n_classes=3)

# --------- Fusion des classes ---------
labels_fused = labels_crf.copy()
labels_fused[labels_crf == 2] = 0  # fusionne classe 2 avec 0
labels_fused[labels_crf == 1] = 1  # garde classe 1

# --------- KMeans (2 clusters) sur les pixels de classe 1 ---------
mask_class_1 = (labels_fused == 1)
gray_flat = gray[mask_class_1].reshape(-1, 1)

if len(gray_flat) > 0:  # éviter les erreurs si aucune classe 1
    kmeans_class1 = KMeans(n_clusters=2, random_state=42)
    kmeans_class1.fit(gray_flat)
    cluster_labels = kmeans_class1.labels_

    # Créer une nouvelle image de labels (par défaut 0 partout)
    refined_labels = labels_fused.copy()
    refined_labels[mask_class_1] = cluster_labels + 1  # pour éviter d’écraser les zéros existants
else:
    refined_labels = labels_fused.copy()


n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
raw_labels = kmeans.labels_.reshape(h, w)

# Réordonner les labels KMeans selon l’intensité moyenne de chaque cluster
cluster_means = []
for i in range(n_clusters):
    cluster_means.append((i, gray[raw_labels == i].mean()))
cluster_means.sort(key=lambda x: x[1])  # ordre croissant
label_mapping = {old: new for new, (old, _) in enumerate(cluster_means)}
labels_kmeans = np.vectorize(label_mapping.get)(raw_labels)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Charger l'image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# --------- KMeans (3 classes) ---------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gray.flatten().reshape(-1, 1))
raw_labels = kmeans.labels_.reshape(h, w)

# Réordonner les labels KMeans selon l’intensité moyenne de chaque cluster
cluster_means = []
for i in range(n_clusters):
    cluster_means.append((i, gray[raw_labels == i].mean()))
cluster_means.sort(key=lambda x: x[1])  # ordre croissant
label_mapping = {old: new for new, (old, _) in enumerate(cluster_means)}
labels_kmeans = np.vectorize(label_mapping.get)(raw_labels)

# --------- CRF sur KMeans (3 classes) ---------
def apply_crf(image_rgb, labels, n_classes, gt_prob=0.7, iter=5):
    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], n_classes)
    unary = unary_from_labels(labels.astype(np.int32), n_classes, gt_prob=gt_prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_rgb, compat=10)
    Q = d.inference(iter)
    refined = np.argmax(Q, axis=0).reshape(labels.shape)
    return refined

labels_crf_before_fusion = apply_crf(image_rgb, labels_kmeans, n_classes=3)

labels_fused = labels_kmeans.copy()
labels_fused[labels_kmeans == 2] = 1
labels_fused[labels_kmeans == 1] = 1

mask_class_0 = (labels_fused == 0).astype(np.uint8)

filled_mask = binary_fill_holes(mask_class_0).astype(np.uint8)

# Reconstruire labels_fused avec masque propre
labels_fused_clean = np.zeros_like(labels_fused)
labels_fused_clean[filled_mask == 1] = 0
labels_fused_clean[filled_mask == 0] = 1

# --------- Détection des contours et ajustement des ellipses ---------
contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image pour afficher les contours
output_image = np.copy(image_rgb)

# Parcourir les contours détectés
for contour in contours:
    if len(contour) >= 5:  # Nombre minimum de points pour ajuster une ellipse
        # Ajuster une ellipse à chaque contour
        ellipse = cv2.fitEllipse(contour)
        
        # Calculer le rapport des axes pour évaluer la circularité
        (center, axes, angle) = ellipse
        ratio = min(axes) / max(axes)  # Plus ce ratio est proche de 1, plus la forme est circulaire
        
        # Si l'ellipse est suffisamment circulaire, colorier en rouge
        if ratio > 0.7:  # Seuil ajustable pour considérer comme circulaire
            cv2.ellipse(output_image, ellipse, (255, 0, 0), 2)  # Dessiner l'ellipse en rouge

# --------- Affichage ---------
plt.figure(figsize=(20, 12))
plt.imshow(output_image)
plt.title('Contours avec ellipses circulaires colorés en rouge')
plt.axis('off')
plt.tight_layout()
plt.show()

from scipy.ndimage import convolve

# ---------- Paramètres ----------
window_size = 5  # Taille de la fenêtre (doit être impair)
threshold = 3    # Nombre de voisins de même classe minimum

# ---------- Préparer le noyau de convolution ----------
kernel = np.ones((window_size, window_size), dtype=np.uint8)
kernel[window_size // 2, window_size // 2] = 0  # Exclure le centre

# ---------- Calculer pour chaque classe séparément ----------
final_mask = np.zeros_like(labels_fused_clean, dtype=np.uint8)

for cls in [0, 1]:
    mask_cls = (labels_fused_clean == cls).astype(np.uint8)
    same_neighbor_count = convolve(mask_cls, kernel, mode='constant', cval=0)
    # Garder les pixels qui sont de cette classe ET ont assez de voisins identiques
    final_mask[np.logical_and(mask_cls == 1, same_neighbor_count >= threshold)] = 1

# ---------- Affichage ----------
plt.figure(figsize=(12, 6))
plt.imshow(final_mask, cmap='gray')
plt.title(f"Pixels avec ≥{threshold} voisins identiques (même classe) dans {window_size}x{window_size}")
plt.axis("off")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_fill_holes

# Fonction pour construire le modèle UNet
def unet(input_size=(512, 512, 1)):  # Adapter la taille d'entrée à 512x512
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.UpSampling2D((2, 2))(u6)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.UpSampling2D((2, 2))(u7)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.UpSampling2D((2, 2))(u8)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.UpSampling2D((2, 2))(u9)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Charger l'image (labelfusedclean) et prétraitement
labelfusedclean = cv2.imread('path_to_labelfusedclean_image.png', cv2.IMREAD_GRAYSCALE)

# Redimensionner l'image à 512x512
labelfusedclean = cv2.resize(labelfusedclean, (512, 512))  # Redimensionner à la taille 512x512
labelfusedclean = np.expand_dims(labelfusedclean, axis=-1)  # Ajouter une dimension pour le canal
labelfusedclean = labelfusedclean / 255.0  # Normalisation

# Ajouter une dimension pour le batch
labelfusedclean = np.expand_dims(labelfusedclean, axis=0)

# Création et entraînement du modèle UNet
model = unet(input_size=(512, 512, 1))  # Adapter la taille d'entrée à 512x512

# Suppose que le modèle est déjà formé, sinon tu devrais charger un modèle pré-entraîné ou entraîner le modèle
# Par exemple : model.load_weights('chemin/vers/poids_du_modele.h5')

# Appliquer le modèle UNet sur l'image
mask = model.predict(labelfusedclean)

# Conversion de la sortie en format d'image
mask = np.squeeze(mask)  # Enlever les dimensions inutiles
mask = (mask > 0.5).astype(np.uint8)  # Seuillage pour obtenir un masque binaire

# Affichage du masque final détecté
plt.figure(figsize=(12, 8))
plt.imshow(mask, cmap='gray')
plt.title('Masque des motifs cohérents détectés')
plt.axis('off')
plt.tight_layout()
plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.morphology import disk, opening, dilation
import os
import json

# --------- Fonctions utilitaires ---------
def apply_crf(image_rgb, labels, n_classes, gt_prob=0.7, iter=5):
    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], n_classes)
    unary = unary_from_labels(labels.astype(np.int32), n_classes, gt_prob=gt_prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_rgb, compat=10)
    Q = d.inference(iter)
    refined = np.argmax(Q, axis=0).reshape(labels.shape)
    return refined

def mask_to_labelme_shapes(mask, class_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.squeeze()
            if len(contour.shape) == 1:
                continue  # Évite les artefacts
            points = contour.tolist()
            shapes.append({
                "label": class_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
    return shapes

def save_labelme_json(json_path, image_filename, image_shape, shapes):
    h, w = image_shape[:2]
    labelme_json = {
        "version": "4.5.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    with open(json_path, 'w') as f:
        json.dump(labelme_json, f, indent=2)

# --------- Traitement principal ---------
def process_image(image_path):
    # Lecture
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(gray.flatten().reshape(-1, 1))
    raw_labels = kmeans.labels_.reshape(h, w)

    # Tri des labels
    cluster_means = [(i, gray[raw_labels == i].mean()) for i in range(3)]
    cluster_means.sort(key=lambda x: x[1])
    label_mapping = {old: new for new, (old, _) in enumerate(cluster_means)}
    labels_kmeans = np.vectorize(label_mapping.get)(raw_labels)

    # CRF
    labels_crf = apply_crf(image_rgb, labels_kmeans, n_classes=3)

    # Fusion des classes 1 et 2
    labels_fused = labels_crf.copy()
    labels_fused[labels_crf == 2] = 1

    # Masques binaires
    mask_class_0 = (labels_fused == 0).astype(np.uint8)
    mask_class_1 = (labels_fused == 1).astype(np.uint8)

    # Morphologie : ouverture + dilatation
    selem = disk(3)
    mask_class_0_clean = dilation(opening(mask_class_0, selem), selem) * 255
    mask_class_1_clean = dilation(opening(mask_class_1, selem), selem) * 255

    # Générer les formes pour LabelMe
    shapes = []
    shapes.extend(mask_to_labelme_shapes(mask_class_0_clean, "class_0"))
    shapes.extend(mask_to_labelme_shapes(mask_class_1_clean, "class_1"))

    # Sauvegarde JSON
    base_name = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    json_path = os.path.join(os.path.dirname(image_path), name_wo_ext + ".json")
    save_labelme_json(json_path, base_name, image.shape, shapes)

    print(f"Annotation sauvegardée : {json_path}")

# --------- Exécution ---------
if __name__ == "__main__":
    # Spécifiez ici votre image ou parcourez un dossier
    image_path = "image.jpg"  # Chemin vers votre image
    process_image(image_path)



import os
import json
import cv2
import numpy as np
from glob import glob

def labelme_to_coco(labelme_dir, output_path, category_mapping=None):
    images = []
    annotations = []
    categories = []
    ann_id = 1
    image_id = 1

    # Créer les catégories
    if category_mapping is None:
        category_mapping = {"class_0": 1, "class_1": 2}
    
    for label, cat_id in category_mapping.items():
        categories.append({"id": cat_id, "name": label})

    labelme_files = glob(os.path.join(labelme_dir, "*.json"))

    for labelme_file in labelme_files:
        with open(labelme_file, 'r') as f:
            data = json.load(f)

        image_filename = data['imagePath']
        img_path = os.path.join(labelme_dir, image_filename)
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {img_path}")
            continue

        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        images.append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        for shape in data['shapes']:
            label = shape['label']
            if label not in category_mapping:
                continue  # Ignore classes non mappées

            category_id = category_mapping[label]
            points = np.array(shape['points'], dtype=np.float32)
            if points.shape[0] < 3:
                continue  # Skip small/invalid polygons

            # Fermer le polygone s'il n'est pas fermé
            if not np.allclose(points[0], points[-1]):
                points = np.vstack([points, points[0]])

            segmentation = [points.flatten().tolist()]
            area = cv2.contourArea(points.astype(np.int32))
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": 0
            })
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved to: {output_path}")

if __name__ == "__main__":
    labelme_dir = "chemin/vers/dossier_avec_json_et_images"
    output_json = "annotations_coco.json"
    # Facultatif : mapping personnalisé LabelMe → COCO category_id
    category_mapping = {"class_0": 1, "class_1": 2}

    labelme_to_coco(labelme_dir, output_json, category_mapping)



import os
import json
import cv2
import numpy as np
from glob import glob
from shapely.geometry import Polygon, box

def labelme_to_coco_split4(labelme_dir, output_path, category_mapping=None):
    images = []
    annotations = []
    categories = []
    ann_id = 1
    image_id = 1

    if category_mapping is None:
        category_mapping = {"class_0": 1, "class_1": 2}

    for label, cat_id in category_mapping.items():
        categories.append({"id": cat_id, "name": label})

    labelme_files = glob(os.path.join(labelme_dir, "*.json"))

    for labelme_file in labelme_files:
        with open(labelme_file, 'r') as f:
            data = json.load(f)

        image_filename = data['imagePath']
        img_path = os.path.join(labelme_dir, image_filename)
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur de lecture : {img_path}")
            continue

        H, W = img.shape[:2]
        h_half, w_half = H // 2, W // 2

        # Définir les 4 régions
        quadrants = [
            ((0, 0), (w_half, h_half)),  # Haut-gauche
            ((w_half, 0), (W, h_half)),  # Haut-droit
            ((0, h_half), (w_half, H)),  # Bas-gauche
            ((w_half, h_half), (W, H))   # Bas-droit
        ]

        for qid, ((x0, y0), (x1, y1)) in enumerate(quadrants):
            quad_box = box(x0, y0, x1, y1)
            new_shapes = []

            for shape in data['shapes']:
                label = shape['label']
                if label not in category_mapping:
                    continue

                points = np.array(shape['points'], dtype=np.float32)
                if len(points) < 3:
                    continue

                polygon = Polygon(points)
                inter = polygon.intersection(quad_box)

                if inter.is_empty or not inter.is_valid or inter.area < 1:
                    continue

                inter_pts = np.array(inter.exterior.coords)
                inter_pts_shifted = inter_pts - np.array([[x0, y0]])

                new_shapes.append({
                    "label": label,
                    "points": inter_pts_shifted.tolist()
                })

            if not new_shapes:
                continue  # pas d'objet dans ce quadrant

            # Sauvegarder nouvelle image (optionnel, sinon juste référencer)
            crop_filename = f"{os.path.splitext(image_filename)[0]}_crop{qid}.png"
            crop_path = os.path.join(labelme_dir, crop_filename)
            cv2.imwrite(crop_path, img[y0:y1, x0:x1])

            images.append({
                "id": image_id,
                "file_name": crop_filename,
                "width": x1 - x0,
                "height": y1 - y0
            })

            for shape in new_shapes:
                points = np.array(shape['points'], dtype=np.float32)
                if len(points) < 3:
                    continue

                segmentation = [points.flatten().tolist()]
                area = cv2.contourArea(points.astype(np.int32))
                x, y, w, h = cv2.boundingRect(points.astype(np.int32))
                category_id = category_mapping[shape['label']]

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0
                })
                ann_id += 1

            image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved to: {output_path}")

# Exemple d’utilisation
if __name__ == "__main__":
    labelme_dir = "chemin/vers/dossier_avec_json_et_images"
    output_json = "annotations_coco_split4.json"
    category_mapping = {"class_0": 1, "class_1": 2}

    labelme_to_coco_split4(labelme_dir, output_json, category_mapping)


import json
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def visualize_coco_annotations(coco_json_path, image_dir, num_images=5):
    # Charger le fichier JSON COCO
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # Lier les annotations aux images
    image_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    for img_data in images[:num_images]:
        img_path = os.path.join(image_dir, img_data['file_name'])
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.set_title(img_data['file_name'])
        ax.axis('off')

        anns = image_to_anns.get(img_data['id'], [])

        for ann in anns:
            segs = ann['segmentation']
            cat_id = ann['category_id']
            label = categories[cat_id]

            for seg in segs:
                poly = np.array(seg).reshape((-1, 2))
                patch = Polygon(poly, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(patch)
                ax.text(poly[0][0], poly[0][1] - 5, label, color='red', fontsize=12, backgroundcolor='white')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    coco_json = "annotations_coco_split4.json"  # ou votre fichier coco
    image_dir = "chemin/vers/les/images"
    visualize_coco_annotations(coco_json, image_dir, num_images=5)


import os
import json
import numpy as np
from PIL import Image
from patchify import patchify
import shutil

# Paramètres
image_path = "chemin/vers/ton_image.png"
json_path = "chemin/vers/ton_image.json"
output_dir = "patches/"
patch_size = (512, 350)  # Taille de chaque patch
overlap = 50

os.makedirs(output_dir, exist_ok=True)

def load_labelme_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def adjust_shape_coordinates(shape, offset_x, offset_y, patch_width, patch_height):
    new_points = []
    for point in shape['points']:
        x, y = point
        if offset_x <= x < offset_x + patch_width and offset_y <= y < offset_y + patch_height:
            new_points.append([x - offset_x, y - offset_y])
        else:
            return None  # La forme est en dehors du patch
    if not new_points:
        return None
    new_shape = shape.copy()
    new_shape['points'] = new_points
    return new_shape

def extract_patches(image, annotations, patch_size, overlap, output_dir):
    width, height = image.size
    pw, ph = patch_size
    step_x = pw - overlap
    step_y = ph - overlap

    patch_id = 0
    for y in range(0, height - ph + 1, step_y):
        for x in range(0, width - pw + 1, step_x):
            patch = image.crop((x, y, x + pw, y + ph))
            patch_filename = f"patch_{patch_id}.png"
            patch.save(os.path.join(output_dir, patch_filename))

            # Filtrer les annotations dans le patch
            new_shapes = []
            for shape in annotations['shapes']:
                adjusted = adjust_shape_coordinates(shape, x, y, pw, ph)
                if adjusted:
                    new_shapes.append(adjusted)

            # Créer nouveau JSON
            patch_json = annotations.copy()
            patch_json['imagePath'] = patch_filename
            patch_json['imageHeight'] = ph
            patch_json['imageWidth'] = pw
            patch_json['shapes'] = new_shapes

            patch_json_path = os.path.join(output_dir, f"patch_{patch_id}.json")
            with open(patch_json_path, 'w') as f:
                json.dump(patch_json, f, indent=2)

            patch_id += 1

# Chargement
image = Image.open(image_path)
annotations = load_labelme_annotations(json_path)

# Traitement
extract_patches(image, annotations, patch_size, overlap, output_dir)


from shapely.geometry import Polygon, box
from shapely.errors import TopologicalError

def adjust_shape_coordinates(shape, offset_x, offset_y, patch_width, patch_height):
    try:
        # Crée le polygone original
        polygon = Polygon(shape['points'])

        # Boîte du patch dans les coordonnées globales
        patch_rect = box(offset_x, offset_y, offset_x + patch_width, offset_y + patch_height)

        # Intersection
        intersected = polygon.intersection(patch_rect)

        if intersected.is_empty:
            return None

        # Convertir l'intersection en liste de points relative au patch
        if intersected.geom_type == 'Polygon':
            points = np.array(intersected.exterior.coords)
        elif intersected.geom_type == 'MultiPolygon':
            # Prendre le plus grand polygone (optionnel, à adapter si tu veux tous les fragments)
            largest = max(intersected.geoms, key=lambda a: a.area)
            points = np.array(largest.exterior.coords)
        else:
            return None

        # Convertir en coordonnées locales au patch
        local_points = [[x - offset_x, y - offset_y] for x, y in points]

        new_shape = shape.copy()
        new_shape['points'] = local_points
        return new_shape
    except TopologicalError:
        return None

Garder patch_size = (512, 350)

Fixer step_x = 256, step_y = 175 → overlap = patch_width - step_x = 256, soit chevauchement de 256 px


import os
import json
from PIL import Image, ImageEnhance, ImageOps

# Paramètres
image_path = "chemin/vers/ton_image.png"
json_path = "chemin/vers/ton_image.json"
output_dir = "patches/"
patch_size = (512, 350)
overlap = 50

os.makedirs(output_dir, exist_ok=True)

def load_labelme_annotations(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def adjust_shape_coordinates(shape, offset_x, offset_y, patch_width, patch_height):
    new_points = []
    for point in shape['points']:
        x, y = point
        if offset_x <= x < offset_x + patch_width and offset_y <= y < offset_y + patch_height:
            new_points.append([x - offset_x, y - offset_y])
        else:
            return None
    if not new_points:
        return None
    new_shape = shape.copy()
    new_shape['points'] = new_points
    return new_shape

def apply_augmentations(image, shapes, base_filename, output_dir, width, height):
    augmentations = {
        "flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "invert": lambda img: ImageOps.invert(img.convert("RGB")),
        "rotate90": lambda img: img.rotate(90, expand=True),
        "rotate180": lambda img: img.rotate(180, expand=True),
        "color": lambda img: ImageEnhance.Color(img).enhance(1.8),
        "sharpness": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        "solarize": lambda img: ImageOps.solarize(img.convert("RGB"), threshold=128)
    }

    for aug_name, aug_fn in augmentations.items():
        aug_img = aug_fn(image)
        aug_filename = f"{base_filename}_{aug_name}.png"
        aug_path = os.path.join(output_dir, aug_filename)
        aug_img.save(aug_path)

        aug_json = {
            "imagePath": aug_filename,
            "imageHeight": aug_img.height,
            "imageWidth": aug_img.width,
            "shapes": shapes  # Même annotations que l’image originale (valide pour les augmentations simples)
        }
        with open(os.path.join(output_dir, f"{base_filename}_{aug_name}.json"), 'w') as f:
            json.dump(aug_json, f, indent=2)

def extract_patches(image, annotations, patch_size, overlap, output_dir):
    width, height = image.size
    pw, ph = patch_size
    step_x = pw - overlap
    step_y = ph - overlap

    patch_id = 0
    for y in range(0, height - ph + 1, step_y):
        for x in range(0, width - pw + 1, step_x):
            if patch_id==0:
                patch = image.crop((x, y, x + pw, y + ph))
            elif patch_id==1:
                patch = image.crop((x, y, x + pw + overlap, y + ph))
            elif patch_id==2:
                patch = image.crop((x, y, x + pw, y + ph+ overlap))
            else:
                patch = image.crop((x, y, x + pw+ overlap, y + ph+ overlap))
            patch = patch.resize((512,400))
            base_filename = f"patch_{patch_id}"
            patch_filename = f"{base_filename}.png"
            patch.save(os.path.join(output_dir, patch_filename))

            new_shapes = []
            for shape in annotations['shapes']:
                adjusted = adjust_shape_coordinates(shape, x, y, pw, ph)
                if adjusted:
                    new_shapes.append(adjusted)

            patch_json = {
                "imagePath": patch_filename,
                "imageHeight": ph,
                "imageWidth": pw,
                "shapes": new_shapes
            }

            with open(os.path.join(output_dir, f"{base_filename}.json"), 'w') as f:
                json.dump(patch_json, f, indent=2)

            apply_augmentations(patch, new_shapes, base_filename, output_dir, pw, ph)

            patch_id += 1

# Chargement et exécution
image = Image.open(image_path)
annotations = load_labelme_annotations(json_path)
extract_patches(image, annotations, patch_size, overlap, output_dir)

import pytorch_lightning as pl
from finetune import SAMFinetuner  # du repo cloné
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_image_encoder", action="store_true")
    args = parser.parse_args()

    model = SAMFinetuner(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        freeze_image_encoder=args.freeze_image_encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.steps,
        data_root=args.data_root
    )

    trainer = pl.Trainer(
        max_steps=args.steps,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        precision=16,
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()

import os
import json
from PIL import Image, ImageEnhance, ImageOps

# Paramètres
image_path = "chemin/vers/ton_image.png"
json_path = "chemin/vers/ton_image.json"
output_dir = "patches/"
patch_size = (512, 350)  # base patch size
resized_size = (512, 400)  # taille après redimensionnement
overlap = 50

os.makedirs(output_dir, exist_ok=True)

def load_labelme_annotations(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def adjust_shape_coordinates(shape, offset_x, offset_y, crop_w, crop_h, resized_w, resized_h):
    scale_x = resized_w / crop_w
    scale_y = resized_h / crop_h

    new_points = []
    for x, y in shape['points']:
        if offset_x <= x < offset_x + crop_w and offset_y <= y < offset_y + crop_h:
            x_local = x - offset_x
            y_local = y - offset_y
            x_resized = x_local * scale_x
            y_resized = y_local * scale_y
            new_points.append([x_resized, y_resized])

    if not new_points:
        return None

    new_shape = shape.copy()
    new_shape['points'] = new_points
    return new_shape

def apply_augmentations(image, shapes, base_filename, output_dir):
    augmentations = {
        "flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "invert": lambda img: ImageOps.invert(img.convert("RGB")),
        "rotate90": lambda img: img.rotate(90, expand=True),
        "rotate180": lambda img: img.rotate(180, expand=True),
        "color": lambda img: ImageEnhance.Color(img).enhance(1.8),
        "sharpness": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        "solarize": lambda img: ImageOps.solarize(img.convert("RGB"), threshold=128)
    }

    for aug_name, aug_fn in augmentations.items():
        aug_img = aug_fn(image)
        aug_filename = f"{base_filename}_{aug_name}.png"
        aug_img.save(os.path.join(output_dir, aug_filename))

        aug_json = {
            "imagePath": aug_filename,
            "imageHeight": aug_img.height,
            "imageWidth": aug_img.width,
            "shapes": shapes
        }
        with open(os.path.join(output_dir, f"{base_filename}_{aug_name}.json"), 'w') as f:
            json.dump(aug_json, f, indent=2)

def extract_patches(image, annotations, patch_size, resized_size, overlap, output_dir):
    width, height = image.size
    pw, ph = patch_size
    rw, rh = resized_size

    patch_id = 0
    for y in range(0, height, ph - overlap):
        for x in range(0, width, pw - overlap):
            # Déterminer la taille du crop en fonction de patch_id
            if patch_id == 0:
                crop_w, crop_h = pw, ph
            elif patch_id == 1:
                crop_w, crop_h = pw + overlap, ph
            elif patch_id == 2:
                crop_w, crop_h = pw, ph + overlap
            else:
                crop_w, crop_h = pw + overlap, ph + overlap

            # Empêcher le dépassement de l'image
            crop_w = min(crop_w, width - x)
            crop_h = min(crop_h, height - y)

            patch = image.crop((x, y, x + crop_w, y + crop_h))
            patch = patch.resize(resized_size)

            base_filename = f"patch_{patch_id}"
            patch_filename = f"{base_filename}.png"
            patch.save(os.path.join(output_dir, patch_filename))

            new_shapes = []
            for shape in annotations['shapes']:
                adjusted = adjust_shape_coordinates(
                    shape, x, y, crop_w, crop_h, rw, rh
                )
                if adjusted:
                    new_shapes.append(adjusted)

            patch_json = {
                "imagePath": patch_filename,
                "imageHeight": rh,
                "imageWidth": rw,
                "shapes": new_shapes
            }

            with open(os.path.join(output_dir, f"{base_filename}.json"), 'w') as f:
                json.dump(patch_json, f, indent=2)

            apply_augmentations(patch, new_shapes, base_filename, output_dir)

            patch_id += 1

# Chargement et exécution
image = Image.open(image_path)
annotations = load_labelme_annotations(json_path)
extract_patches(image, annotations, patch_size, resized_size, overlap, output_dir)



def transform_shapes(shapes, aug_name, img_width, img_height):
    transformed_shapes = []
    for shape in shapes:
        new_shape = shape.copy()
        new_points = []
        for x, y in shape['points']:
            if aug_name == "flip":
                # flip horizontal (gauche-droite)
                x_new = img_width - x
                y_new = y
            elif aug_name == "rotate90":
                # rotation 90° clockwise
                x_new = y
                y_new = img_width - x
            elif aug_name == "rotate180":
                # rotation 180°
                x_new = img_width - x
                y_new = img_height - y
            else:
                x_new, y_new = x, y
            new_points.append([x_new, y_new])
        new_shape['points'] = new_points
        transformed_shapes.append(new_shape)
    return transformed_shapes

def apply_augmentations(image, shapes, base_filename, output_dir):
    augmentations = {
        "flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "invert": lambda img: ImageOps.invert(img.convert("RGB")),
        "rotate90": lambda img: img.rotate(90, expand=True),
        "rotate180": lambda img: img.rotate(180, expand=True),
        "color": lambda img: ImageEnhance.Color(img).enhance(1.8),
        "sharpness": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        "solarize": lambda img: ImageOps.solarize(img.convert("RGB"), threshold=128)
    }

    for aug_name, aug_fn in augmentations.items():
        aug_img = aug_fn(image)
        aug_filename = f"{base_filename}_{aug_name}.png"
        aug_img.save(os.path.join(output_dir, aug_filename))

        # Pour flip et rotations, appliquer la transformation sur les annotations
        if aug_name in ["flip", "rotate90", "rotate180"]:
            transformed_shapes = transform_shapes(
                shapes, aug_name, image.width, image.height
            )
        else:
            transformed_shapes = shapes

        aug_json = {
            "imagePath": aug_filename,
            "imageHeight": aug_img.height,
            "imageWidth": aug_img.width,
            "shapes": transformed_shapes
        }
        with open(os.path.join(output_dir, f"{base_filename}_{aug_name}.json"), 'w') as f:
            json.dump(aug_json, f, indent=2)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import perimeter
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# === CONFIGURATION DU MODELE ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = "path/to/your/model_final.pth"  # ← Remplacer par ton chemin

predictor = DefaultPredictor(cfg)

# === CHARGER L’IMAGE ===
image_path = "path/to/your/image.jpg"  # ← Remplacer par ton image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === INFERENCE ===
outputs = predictor(image)
instances = outputs["instances"].to("cpu")

masks = instances.pred_masks.numpy()
classes = instances.pred_classes.numpy()
scores = instances.scores.numpy()

# === AFFICHER LES DETECTIONS ===
metadata = MetadataCatalog.get("__unused")
metadata.set(thing_classes=["Classe A", "Classe B", "Classe C"])
v = Visualizer(image_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
v = v.draw_instance_predictions(instances)
cv2.imshow("Détections", cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# === EXTRACTION DES DONNEES PAR OBJET ===
perimeters = []
areas = []
means = []
stds = []
class_ids = []

for i, mask in enumerate(masks):
    # Aire
    area = np.sum(mask)
    # Périmètre (scikit-image)
    perim = perimeter(mask)
    # Moyenne et écart-type (en niveaux de gris)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask]
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)

    # Stockage
    perimeters.append(perim)
    areas.append(area)
    means.append(mean_val)
    stds.append(std_val)
    class_ids.append(classes[i])

# === COULEURS PAR CLASSE ===
colors = ['red', 'green', 'blue']
color_map = [colors[c] for c in class_ids]

# === NUAGE 1 : Périmètre vs Aire ===
plt.figure(figsize=(6, 5))
plt.scatter(perimeters, areas, c=color_map)
plt.xlabel("Périmètre")
plt.ylabel("Aire")
plt.title("Nuage 1 : Périmètre vs Aire")
plt.grid(True)

# === NUAGE 2 : Moyenne vs Écart-type ===
plt.figure(figsize=(6, 5))
plt.scatter(means, stds, c=color_map)
plt.xlabel("Moyenne intensité")
plt.ylabel("Écart-type intensité")
plt.title("Nuage 2 : Moyenne vs Écart-type")
plt.grid(True)

# === NUAGE 3 : Aire vs Moyenne (ou re-Périmètre si souhaité) ===
plt.figure(figsize=(6, 5))
plt.scatter(areas, means, c=color_map)
plt.xlabel("Aire")
plt.ylabel("Moyenne intensité")
plt.title("Nuage 3 : Aire vs Moyenne intensité")
plt.grid(True)

plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import perimeter
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# === CONFIG DETECTRON2 ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = "path/to/your/model_final.pth"  # ← mettre le chemin vers ton modèle entraîné

predictor = DefaultPredictor(cfg)

# === IMAGE ===
image_path = "path/to/your/image.jpg"  # ← chemin de ton image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === INFERENCE ===
outputs = predictor(image)
instances = outputs["instances"].to("cpu")
masks = instances.pred_masks.numpy()
classes = instances.pred_classes.numpy()

# === VISUALISATION DES DETECTIONS ===
metadata = MetadataCatalog.get("__unused")
metadata.set(thing_classes=["Classe A", "Classe B", "Classe C"])
v = Visualizer(image_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
v = v.draw_instance_predictions(instances)
cv2.imshow("Détections", cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# === EXTRACTION DES DONNÉES ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

perimeters, areas, means, stds, class_ids = [], [], [], [], []

for i, mask in enumerate(masks):
    area = np.sum(mask)
    perim = perimeter(mask)
    pixels = gray[mask]
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)

    areas.append(area)
    perimeters.append(perim)
    means.append(mean_val)
    stds.append(std_val)
    class_ids.append(classes[i])

# === COULEUR PAR CLASSE ===
colors = ['red', 'green', 'blue']
color_map = [colors[c] for c in class_ids]
x_ids = list(range(len(class_ids)))

# === NUAGE 1 : AIRE ===
plt.figure()
plt.scatter(x_ids, areas, c=color_map)
plt.xlabel("Objet")
plt.ylabel("Aire")
plt.title("Aire des objets détectés (couleur = classe)")
plt.grid(True)

# === NUAGE 2 : PÉRIMÈTRE ===
plt.figure()
plt.scatter(x_ids, perimeters, c=color_map)
plt.xlabel("Objet")
plt.ylabel("Périmètre")
plt.title("Périmètre des objets détectés (couleur = classe)")
plt.grid(True)

# === NUAGE 3 : MOYENNE INTENSITÉ ===
plt.figure()
plt.scatter(x_ids, means, c=color_map)
plt.xlabel("Objet")
plt.ylabel("Moyenne intensité (gris)")
plt.title("Moyenne d’intensité par objet (couleur = classe)")
plt.grid(True)

# === NUAGE 4 : ÉCART-TYPE INTENSITÉ ===
plt.figure()
plt.scatter(x_ids, stds, c=color_map)
plt.xlabel("Objet")
plt.ylabel("Écart-type intensité (gris)")
plt.title("Écart-type d’intensité par objet (couleur = classe)")
plt.grid(True)

plt.show()

import os
import json
from collections import defaultdict
from tqdm import tqdm

def coco_to_labelme(coco_json_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id → image info
    images = {img["id"]: img for img in coco["images"]}
    # Map category_id → category name
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Group annotations by image
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for image_id, anns in tqdm(ann_by_image.items(), desc="Conversion en cours"):
        image_info = images[image_id]
        filename = image_info["file_name"]

        labelme_shapes = []

        for ann in anns:
            cat_name = categories[ann["category_id"]]

            # Utiliser segmentation (assume format polygon)
            segs = ann.get("segmentation", [])
            if not isinstance(segs, list):
                continue  # Ignore RLE ou formats non pris en charge ici

            for seg in segs:
                # segmentation est une liste plate → on transforme en [[x1, y1], [x2, y2], ...]
                points = [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)]

                shape = {
                    "label": cat_name,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                labelme_shapes.append(shape)

        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": labelme_shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": image_info["height"],
            "imageWidth": image_info["width"]
        }

        # Sauvegarder le fichier .json dans le style LabelMe
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
        with open(output_path, 'w') as f:
            json.dump(labelme_json, f, indent=4)

    print(f"✅ Conversion terminée. JSON LabelMe enregistrés dans : {output_dir}")

# 🔧 Exemple d’utilisation :
coco_json_path = "/chemin/vers/annotations_coco.json"
image_dir = "/chemin/vers/images"
output_dir = "/chemin/vers/labelme_jsons"

coco_to_labelme(coco_json_path, image_dir, output_dir)


import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def labelme_to_coco_detectron2(labelme_dir, output_json):
    images = []
    annotations = []
    categories = []
    label_to_id = {}

    image_id = 1
    ann_id = 1

    json_files = glob(os.path.join(labelme_dir, "*.json"))

    for json_file in tqdm(json_files, desc="Conversion en COCO"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_filename = data["imagePath"]
        image_height = data["imageHeight"]
        image_width = data["imageWidth"]

        images.append({
            "id": image_id,
            "file_name": image_filename,
            "height": image_height,
            "width": image_width
        })

        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape.get("shape_type", "polygon")

            if shape_type != "polygon":
                continue  # on ignore les rectangles, cercles, etc. pour segmentation instance

            # Enregistrer la catégorie si inconnue
            if label not in label_to_id:
                label_id = len(label_to_id) + 1
                label_to_id[label] = label_id
                categories.append({
                    "id": label_id,
                    "name": label,
                    "supercategory": "object"
                })

            # Créer segmentation
            segmentation = [np.array(points).flatten().tolist()]

            # Masque pour calculer bbox et area
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            polygon = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 1)

            area = float(np.sum(mask))
            x, y, w, h = cv2.boundingRect(polygon[0])
            bbox = [x, y, w, h]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": label_to_id[label],
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

        image_id += 1

    coco_output = {
        "info": {
            "description": "LabelMe converted to COCO",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"\n✅ Fichier COCO généré : {output_json}")

# 🔧 Exemple d’utilisation
labelme_dir = "/chemin/vers/dossier_labelme"
output_json = "annotations_coco.json"

labelme_to_coco_detectron2(labelme_dir, output_json)


import os
import json
import cv2
import random
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from shutil import copyfile

# === Config ===
INPUT_IMAGE_DIR = 'images/'               # Dossier contenant les images originales
ANNOTATION_PATH = 'coco.json'             # Fichier COCO original
OUTPUT_ANNOTATION_PATH = 'coco_augmented.json'
AUG_SUFFIXES = ['_flipH', '_flipV', '_rot90', '_rot180', '_rot270']

# === Charger les annotations COCO ===
with open(ANNOTATION_PATH) as f:
    coco_data = json.load(f)

new_images = []
new_annotations = []
new_image_id = max(img['id'] for img in coco_data['images']) + 1
new_annotation_id = max(ann['id'] for ann in coco_data['annotations']) + 1

coco = COCO(ANNOTATION_PATH)
image_id_to_annotations = {}

# Regrouper les annotations par image
for ann in coco_data['annotations']:
    image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

# === Fonctions de transformation ===

def transform_annotation(bbox, img_w, img_h, operation):
    x, y, w, h = bbox
    if operation == 'flipH':
        return [img_w - x - w, y, w, h]
    elif operation == 'flipV':
        return [x, img_h - y - h, w, h]
    elif operation == 'rot90':
        return [y, img_w - x - w, h, w]
    elif operation == 'rot180':
        return [img_w - x - w, img_h - y - h, w, h]
    elif operation == 'rot270':
        return [img_h - y - h, x, h, w]
    return bbox

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def augment_image(img, operation):
    if operation == 'flipH':
        return cv2.flip(img, 1)
    elif operation == 'flipV':
        return cv2.flip(img, 0)
    elif operation == 'rot90':
        return rotate_image(img, 90)
    elif operation == 'rot180':
        return rotate_image(img, 180)
    elif operation == 'rot270':
        return rotate_image(img, 270)
    return img

# === Processus principal ===
for img_info in tqdm(coco_data['images'], desc="Augmenting images"):
    file_name = img_info['file_name']
    image_path = os.path.join(INPUT_IMAGE_DIR, file_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    original_image = cv2.imread(image_path)
    original_h, original_w = original_image.shape[:2]
    annotations = image_id_to_annotations.get(img_info['id'], [])

    for suffix in AUG_SUFFIXES:
        operation = suffix[1:]  # remove '_'

        aug_image = augment_image(original_image.copy(), operation)

        if 'rot' in operation:
            aug_h, aug_w = aug_image.shape[:2]
        else:
            aug_h, aug_w = original_h, original_w

        # Nouveau nom de fichier
        new_file_name = f"{os.path.splitext(file_name)[0]}{suffix}.jpg"
        new_path = os.path.join(INPUT_IMAGE_DIR, new_file_name)
        cv2.imwrite(new_path, aug_image)

        # Ajouter image dans la structure COCO
        new_images.append({
            "id": new_image_id,
            "file_name": new_file_name,
            "height": aug_h,
            "width": aug_w
        })

        # Modifier les annotations pour cette image
        for ann in annotations:
            new_bbox = transform_annotation(ann['bbox'], original_w, original_h, operation)

            new_annotations.append({
                "id": new_annotation_id,
                "image_id": new_image_id,
                "category_id": ann['category_id'],
                "bbox": new_bbox,
                "area": new_bbox[2] * new_bbox[3],
                "iscrowd": ann.get("iscrowd", 0)
            })

            new_annotation_id += 1

        new_image_id += 1

# === Ajouter les nouvelles données au JSON COCO ===
coco_data['images'].extend(new_images)
coco_data['annotations'].extend(new_annotations)

# === Sauvegarde ===
with open(OUTPUT_ANNOTATION_PATH, 'w') as f:
    json.dump(coco_data, f)

print(f"\n✅ Augmentation terminée. Nouveau fichier COCO : {OUTPUT_ANNOTATION_PATH}")


Merci pour le fichier Excel.

En consultant les données, j’ai vu que nous avons uniquement les indices pour chaque composant. Serait-il possible d’avoir les temps de rétention de référence correspondants ?
Dans mes données, je n’ai que les temps de rétention pour les pics.

Existe-t-il une formule pour passer de l’indice au temps de rétention ?

Merci par avance,
import os
import json
from PIL import Image, ImageEnhance, ImageOps
from pycocotools.coco import COCO
from tqdm import tqdm

# === Augmentations définies ===
def apply_augmentations(image, base_filename, output_dir):
    augmentations = {
        "flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "invert": lambda img: ImageOps.invert(img.convert("RGB")),
        "rotate90": lambda img: img.rotate(90, expand=True),
        "rotate180": lambda img: img.rotate(180, expand=True),
        "rotate270": lambda img: img.rotate(270, expand=True),  # ✅ Ajout ici
        "color": lambda img: ImageEnhance.Color(img).enhance(1.8),
        "sharpness": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        "solarize": lambda img: ImageOps.solarize(img.convert("RGB"), threshold=128)
    }

    augmented = []
    for name, aug in augmentations.items():
        new_img = aug(image.copy())
        new_filename = f"{base_filename}_{name}.jpg"
        new_path = os.path.join(output_dir, new_filename)
        new_img.save(new_path)
        augmented.append((name, new_filename, new_img.size))
    return augmented

# === Transformation du polygone ===
def transform_polygon(segmentation, orig_size, transform_name):
    orig_w, orig_h = orig_size
    new_segmentation = []

    for poly in segmentation:
        new_poly = []
        for i in range(0, len(poly), 2):
            x, y = poly[i], poly[i + 1]

            if transform_name == "flip":
                x = orig_w - x
            elif transform_name == "rotate90":
                x, y = y, orig_w - x
            elif transform_name == "rotate180":
                x, y = orig_w - x, orig_h - y
            elif transform_name == "rotate270":
                x, y = orig_h - y, x

            new_poly.extend([x, y])
        new_segmentation.append(new_poly)
    return new_segmentation

# === Calcul de la bbox à partir du polygone ===
def polygon_to_bbox(segmentation):
    x_coords = []
    y_coords = []
    for poly in segmentation:
        x_coords.extend(poly[::2])
        y_coords.extend(poly[1::2])
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# === Calcul de l'aire du polygone ===
def polygon_area(segmentation):
    area = 0
    for poly in segmentation:
        x = poly[::2]
        y = poly[1::2]
        n = len(x)
        a = sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
        area += abs(a) / 2
    return area

# === Chemins ===
IMAGE_DIR = "images/"
ANNOTATION_PATH = "coco.json"
OUTPUT_JSON_PATH = "coco_augmented.json"

# === Charger le JSON COCO ===
with open(ANNOTATION_PATH, 'r') as f:
    coco_data = json.load(f)

coco = COCO(ANNOTATION_PATH)
new_images = []
new_annotations = []
new_image_id = max(img['id'] for img in coco_data['images']) + 1
new_ann_id = max(ann['id'] for ann in coco_data['annotations']) + 1

# === Préparer les annotations groupées par image ===
annotations_by_image = {}
for ann in coco_data['annotations']:
    annotations_by_image.setdefault(ann['image_id'], []).append(ann)

# === Processus principal ===
for img_info in tqdm(coco_data['images'], desc="Augmenting"):
    img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    base_name = os.path.splitext(img_info['file_name'])[0]
    ann_list = annotations_by_image.get(img_info['id'], [])

    augmented_versions = apply_augmentations(img, base_name, IMAGE_DIR)

    for aug_name, new_filename, (new_w, new_h) in augmented_versions:
        new_images.append({
            "id": new_image_id,
            "file_name": new_filename,
            "width": new_w,
            "height": new_h
        })

        for ann in ann_list:
            # ✅ Appliquer transformation si géométrique
            new_segmentation = transform_polygon(
                ann['segmentation'],
                (img.width, img.height),
                aug_name
            ) if aug_name in ['flip', 'rotate90', 'rotate180', 'rotate270'] else ann['segmentation']

            new_bbox = polygon_to_bbox(new_segmentation)
            area = polygon_area(new_segmentation)

            new_annotations.append({
                "id": new_ann_id,
                "image_id": new_image_id,
                "category_id": ann['category_id'],
                "segmentation": new_segmentation,
                "bbox": new_bbox,
                "area": area,
                "iscrowd": ann.get("iscrowd", 0)
            })
            new_ann_id += 1

        new_image_id += 1

# === Mise à jour du fichier COCO ===
coco_data['images'].extend(new_images)
coco_data['annotations'].extend(new_annotations)

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(coco_data, f)

print(f"\n✅ Terminé. Nouveau fichier COCO : {OUTPUT_JSON_PATH}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# === Configuration ===
image_path = "votre_image.jpg"  # Chemin vers votre image
resolution_nm_per_pixel = 0.06  # nm/pixel

# === Setup Detectron2 Predictor ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)

# === Charger l'image ===
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Prédiction ===
outputs = predictor(image)
instances = outputs["instances"].to("cpu")
masks = instances.pred_masks.numpy()

# === Visualisation des masques ===
v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
out = v.draw_instance_predictions(instances)

plt.figure(figsize=(12, 6))
plt.imshow(out.get_image())
plt.title("Objets détectés et segmentés")
plt.axis("off")
plt.show()

# === Calcul du diamètre équivalent et de la surface ===
diameters_nm = []
areas_nm2 = []

for mask in masks:
    area_pixels = np.sum(mask)
    if area_pixels == 0:
        continue
    area_nm2 = area_pixels * (resolution_nm_per_pixel ** 2)
    diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)
    
    diameters_nm.append(diameter_nm)
    areas_nm2.append(area_nm2)

# === Affichage des histogrammes normalisés ===
def plot_histogram(data, xlabel, title, bins=20):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='teal', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Fréquence normalisée")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_histogram(diameters_nm, "Diamètre (nm)", "Histogramme normalisé des diamètres")
plot_histogram(areas_nm2, "Surface (nm²)", "Histogramme normalisé des surfaces")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Button, Label, Entry, Frame
from PIL import Image, ImageTk
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Setup Detectron2 Predictor ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

class ParticleAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyse de Particules - Detectron2")

        self.image_path = None
        self.resolution = 0.06  # Valeur par défaut

        self.setup_ui()

    def setup_ui(self):
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        Button(control_frame, text="Charger Image", command=self.load_image).pack(side="left", padx=5)

        Label(control_frame, text="Résolution (nm/pixel):").pack(side="left")
        self.res_entry = Entry(control_frame, width=10)
        self.res_entry.insert(0, str(self.resolution))
        self.res_entry.pack(side="left", padx=5)

        Button(control_frame, text="Lancer l'Inference", command=self.run_inference).pack(side="left", padx=5)

        # Section image
        self.img_label = Label(self.root)
        self.img_label.pack()

        # Section histogrammes
        self.hist_frame = Frame(self.root)
        self.hist_frame.pack()

        self.hist_canvas1 = None
        self.hist_canvas2 = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path).resize((512, 512))
            self.tk_image = ImageTk.PhotoImage(image)
            self.img_label.config(image=self.tk_image)

    def run_inference(self):
        if not self.image_path:
            return

        try:
            self.resolution = float(self.res_entry.get())
        except ValueError:
            print("Résolution invalide.")
            return

        image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()

        # Visualisation
        v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(instances)
        vis_image = out.get_image()

        # Affichage dans interface
        vis_pil = Image.fromarray(vis_image).resize((512, 512))
        self.tk_image = ImageTk.PhotoImage(vis_pil)
        self.img_label.config(image=self.tk_image)

        # Calcul des métriques
        diameters_nm = []
        areas_nm2 = []

        for mask in masks:
            area_pixels = np.sum(mask)
            if area_pixels == 0:
                continue
            area_nm2 = area_pixels * (self.resolution ** 2)
            diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)

            diameters_nm.append(diameter_nm)
            areas_nm2.append(area_nm2)

        # Affichage des histogrammes dans interface
        self.plot_histogram(diameters_nm, "Diamètre (nm)", "Histogramme des diamètres", canvas_index=1)
        self.plot_histogram(areas_nm2, "Surface (nm²)", "Histogramme des surfaces", canvas_index=2)

    def plot_histogram(self, data, xlabel, title, bins=20, canvas_index=1):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(data, bins=bins, density=True, alpha=0.7, color='teal', edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fréquence normalisée")
        ax.set_title(title)
        ax.grid(True)
        fig.tight_layout()

        # Dessin dans l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas.draw()

        # Suppression de l'ancien histogramme si présent
        if canvas_index == 1:
            if self.hist_canvas1:
                self.hist_canvas1.get_tk_widget().destroy()
            self.hist_canvas1 = canvas
            canvas_widget.grid(row=0, column=0)
        else:
            if self.hist_canvas2:
                self.hist_canvas2.get_tk_widget().destroy()
            self.hist_canvas2 = canvas
            canvas_widget.grid(row=0, column=1)

# === Lancement Interface ===
if __name__ == "__main__":
    root = Tk()
    app = ParticleAnalyzerApp(root)
    root.mainloop()

def plot_histogram(self, data, xlabel, title, bins=20, canvas_index=1):
    fig, ax = plt.subplots(figsize=(4, 3))

    # Histogramme brut
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Fréquence normalisée en nombre
    total = np.sum(counts)
    frequencies = counts / total  # Normalisation manuelle

    # Positions des barres au centre des intervalles
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Affichage en barres
    ax.bar(bin_centers, frequencies, width=(bin_edges[1] - bin_edges[0]), color='teal', edgecolor='black', alpha=0.8)

    # Configuration des axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fréquence normalisée")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()

    # Affichage dans interface
    canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas.draw()

    if canvas_index == 1:
        if self.hist_canvas1:
            self.hist_canvas1.get_tk_widget().destroy()
        self.hist_canvas1 = canvas
        canvas_widget.grid(row=0, column=0)
    else:
        if self.hist_canvas2:
            self.hist_canvas2.get_tk_widget().destroy()
        self.hist_canvas2 = canvas
        canvas_widget.grid(row=0, column=1)
