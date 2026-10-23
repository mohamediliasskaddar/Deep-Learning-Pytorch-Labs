# Lab 2 : Vision par Ordinateur (Classification, Détection & Vision Transformers)

---

## 1. Partie 1 : Classification de Chiffres (MNIST) & Détection d'Objets avec Faster R-CNN

### 1.1 Classification de Chiffres (MNIST) :  Analyse des Résultats (CNN Custom vs AlexNet Fine-Tuned)

Sur la base du tableau comparatif et des métriques obtenues :

- **Custom CNN (From Scratch) :**
  - **Précision (Test Accuracy) :** 98.97%
  - **Min Test Loss :** 0.025816
  - **Temps de calcul :** 132.16 secondes ($\approx 2.20$ minutes)
  - **Taille d'entrée :** $28 \times 28$ (Grayscale)

- **AlexNet (Fine-Tuned) :**
  - **Précision (Test Accuracy) :** 98.85%
  - **Min Test Loss :** 0.022289
  - **Temps de calcul :** 353.20 secondes ($\approx 5.89$ minutes)
  - **Taille d'entrée :** $224 \times 224$ (RGB)

### Interprétation Métier & Bilan Pratique :
1. **Rendement et pertinence de l'architecture :** Bien que la perte minimale d'AlexNet soit légèrement plus basse (0.0222 vs 0.0258), le modèle **CNN Custom** obtient une précision supérieure sur le jeu de test (98.97% contre 98.85%).
2. **Coût de calcul (Overhead computational) :** AlexNet requiert près de **2.7 fois plus de temps GPU** (353s contre 132s). Cet écart s'explique par la nécessité de redimensionner chaque image de $28 \times 28$ vers $224 \times 224$ pixels et de dupliquer les canaux en 3 dimensions RGB pour satisfaire l'entrée du modèle pré-entraîné sur ImageNet.
3. **Conclusion pour la classification simple :** Pour des tâches simples et à petite échelle comme MNIST, construire une architecture légère *from scratch* est nettement plus efficace et économique qu'adapter une grande architecture complexe conçue pour des images haute résolution.

---

### 1.2  Détection d'Objets avec Faster R-CNN : Analyse des Résultats (Penn-Fudan Pedestrian Dataset)

1. **Convergence Rapide :**
   - **Époque 1 :** Loss = 0.2757
   - **Époque 3 :** Loss = 0.0992 (réduction de plus de 64%)
   - **Temps d'entraînement :** 106.17 secondes sur GPU T4.

2. **Performance Visuelle de Localisation et Classification :**
   - **Image Test #1 :** 7 piétons détectés avec une confiance allant jusqu'à 99.6%. Les personnes en arrière-plan et partiellement masquées sont correctement délimitées.
   - **Image Test #2 :** 3 piétons isolés repérés sans aucun faux positif avec des scores de confiance de 99.2% à 99.6%.
   - **Image Test #3 :** 8 piétons détectés dans une scène dense de passage piéton (gestion efficace de l'occlusion partielle).

---
## 2. Partie 2 : Vision Transformer (ViT) construit From Scratch

### Architecture et Implémentation :
Le Vision Transformer (ViT) découpe l'image $28 \times 28$ en $7 \times 7 = 49$ patchs de taille $4 \times 4$. Chaque patch est projeté linéairement sous forme de vecteur d'embedding, auquel sont ajoutés :
- Un jeton de classification **`[CLS]`** apprenable pour capturer l'information globale.
- Des **Positional Embeddings** pour conserver l'information spatiale.
- Un bloc d'encodeur basé sur le mécanisme d'**Auto-Attention Multi-Têtes (Multi-Head Self-Attention - MSA)**.

### Analyse des Performances sur MNIST :
- **Convergence :** La perte d'entraînement baisse progressivement tout au long des 5 époques.
- **Précision obtenue :** La précision atteint environ ~75% à 85% lors d'un entraînement court à partir de zéro (*from scratch*).
- **Temps d'exécution :** ~120 secondes sur GPU T4.

### Comparaison ViT vs CNN :
- **Absence de Biais Inductif (Inductive Bias) :** Contrairement aux CNNs qui possèdent nativement des propriétés de *localité* (pixels voisins) et d'*invariance par translation*, le ViT ne connaît aucune structure spatiale à priori. Il doit tout apprendre à partir des données.
- **Dépendance au volume de données :** Entraîné *from scratch* sur un petit jeu de données (60 000 images simples), un petit ViT sous-performe par rapport à un simple CNN (85% vs 98.97%). Les Vision Transformers nécessitent un pré-entraînement sur d'immenses jeux de données (ImageNet-21k, JFT-300M) pour surpasser les CNNs.

---


## 4. Tests d'Inférence sur Images Personnalisées (Données Réelles)

Pour valider la robustesse des modèles appris, deux tests sur des données utilisateur inédites ont été menés :

1. **Test du CNN Custom sur un chiffre manuscrit externe :**
   - **Découverte majeure :** L'application directe d'un chiffre noir sur fond blanc a d'abord échoué (prédiction erronée). 
   - **Résolution :** L'inversion des couleurs (`ImageOps.invert`) pour aligner la donnée sur la convention MNIST (chiffre blanc sur fond noir) a permis au modèle d'identifier correctement le chiffre avec plus de 90% de confiance.
2. **Test de Faster R-CNN sur une photo de rue externe :**
   - Le modèle pré-entraîné et affiné a réussi à isoler et tracer des boîtes englobantes autour des piétons sur des images réelles téléchargées hors dataset.

---

## 5. Synthèse & Enseignements Clés du Lab

- **L'importance critique du Prétraitement des Données (Data Preprocessing) :** Un modèle ultra-performant peut échouer sur une image simple si le format (fond noir/blanc, normalisation, dimensions) ne correspond pas exactement au format de son entraînement.
- **Adéquation Modèle / Tâche :** Un modèle plus lourd ou plus récent (AlexNet, ViT) n'est pas systématiquement le meilleur choix pour des données simples. Un CNN léger reste le roi de l'efficacité pour MNIST.
- **Classification vs Détection :** 
  - La **classification** attribue une étiquette globale à l'ensemble d'une image.
  - La **détection** (Faster R-CNN) combine la **localisation par régression** (boîtes englobantes) et la **classification** sur plusieurs objets au sein d'une même scène.
- **Mécanisme d'Attention vs Convolutions :** Les CNNs extraient des caractéristiques locales de manière hiérarchique, tandis que les Transformers calculent des interactions globales entre tous les patchs d'une image simultanément.
- **Efficacité du Transfer Learning :** Réutiliser des squelettes pré-entraînés (*ResNet-50-FPN*) permet de résoudre des tâches complexes de détection en moins de 2 minutes de calcul.