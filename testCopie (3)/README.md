# Park Access with AI
---
# Demo


## Description
Ce projet est réalisé dans le cadre du projet M2 du DP IA de l'ISEN Brest par Karim CHEHADE, Tristan SAEZ, Maël CREISMEAS, Céline DUSSUELLE, Mathieu NICOLAS et Océane CRAS.

Il consiste à utiliser divers modèles d'Intelligence Artificielle afin de créer un accès automatisé à un parking en détectant la plaque d'immatriculation et en comparant le modèle de voiture entrant à une base de données.

## Modèles IA
Afin de détecter la plaque d'immatriculation et de comparer les modèles autorisés à rentrer, nous nous sommes basés sur deux modèles pré-entraînés.
### YoloV11
Le modèle YOLO (You Only Look Once) a été utilisé danse ce projet pour plusieurs de ses caractérisiques lui donnant un avantage comparé à d’autres modèles sur les tâches demandés. Notamment le traitement de l’image rapide. La version utilisé v11, a été utilisé pour avoir les dernière mis-à-jour sur ce modèle.
Le modèle Yolov11, permet dans le carde de ce projet, d’effectuer deux tâches distinctes.
- La classification (et l’isolement des objets classifié), quand une voiture entre dans le champ de vision de la caméra, du véhicule et de sa plaque sur une image.
- La détectection et l'identification des caractères sur la plaque d’immatriculation du véhicule
### ResNet-50 IBN
Le modèle ResNet-50 IBN est une architecture à 50 couches avec des blocs résiduels. Il s'agit d'un modèle amélioré de ResNet-50, qui gère mieux les variations d'éclairage. Il sert de backbone au modèle de réidentification des véhicules et agit en tant qu'extracteur de caractéristiques pour la classification des images

Le modèle a été entraîné sur la base de données VRIC qui contient plus de 60 000 images de voitures. 

Le modèle extrait les caractéristiques de l'image de requête et de chacune des images la galerie afin de calculer des scores de similarité. Le véhicule est identifié avec l'image la plus similaire à condition que le score soit supérieur à 0.7.

---
## Lancer le programme

###  Requirements
Afin d'installer les librairies essentielles au projet, il suffit de lancer dans le terminal :
```
pip install -r requirements.txt
```

### Replacer les poids des modèles entraînés
Une fois les librairies installées, il faut replacer les poids des modèles IA. Avec les poids fournis, il suffit de placer le fichier ```net_29.pth``` dans le directory :
```veid/model/result```
et les fichiers ```best.pt``` et ```last.pt``` dans les directories :
```/models/yolov11/results/40_epochs_yolox/weights```
et
 ```/models/yolov11/results/40_epochs_full_yolol/weights```

### Simulation de la caméra
Puisque c'est l'API qui simule la caméra, il faut d'abord run ```app.py``` situé dans ```/web``` afin de lancer le site puis ```model_fct.py``` situé dans ```/ai``` afin de lancer l'IA en backend.

### Site web
Afin d'accéder au site web, un lien vers le localhost est affiché dans le terminal au lancement de ```app.py```.

Le site permet de :
- Détecter d'une voiture entrante
- Vérifier si le véhicule présenté est présent dans la base de donnée des véhicules déjà autorisé à la date actuel.
- Ajouter une voiture dans la base de donnée avec un formulaire.