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


output_image = image_rgb.copy()
num_labels, labels_conn = cv2.connectedComponents(closed_mask)

for i in range(1, num_labels):  # ignorer le fond
    mask = (labels_conn == i).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity > 0.75:  # seuil ajustable
        cv2.drawContours(output_image, [cnt], -1, (255, 0, 0), 2)  # rouge (RGB)

# --------- Affichage ---------
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title('Image originale')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(labels_kmeans, cmap=ListedColormap(plt.cm.tab10.colors[:3]))
plt.title('KMeans (réordonné)')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(labels_crf_before_fusion, cmap=ListedColormap(plt.cm.tab10.colors[:3]))
plt.title('CRF avant fusion (3 classes)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(labels_fused, cmap=ListedColormap(plt.cm.tab10.colors[:2]))
plt.title('Labels fusionnés (avant fermeture)')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(labels_fused_clean, cmap=ListedColormap(plt.cm.tab10.colors[:2]))
plt.title('Labels après fermeture')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(distance, cmap='jet')
plt.title('Carte de distance (watershed)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(labels_watershed, cmap='nipy_spectral')
plt.title('Résultat Watershed')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(output_image)
plt.title('Formes circulaires de classe 1 en rouge')
plt.axis('off')

plt.tight_layout()
plt.show()