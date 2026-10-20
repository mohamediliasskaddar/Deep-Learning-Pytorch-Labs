# Compte-Rendu & Analyse : Lab 2 - Vision par Ordinateur (Classification & Détection d'Objets)

---

## 1. Partie 1 : Classification de Chiffres (MNIST)

### Analyse des Résultats (CNN Custom vs AlexNet Fine-Tuned)

Sur la base du tableau comparatif et du graphique générés :

* **Custom CNN (From Scratch) :**
  * **Précision (Test Accuracy) :** $98.97\%$
  * **Min Test Loss :** $0.025816$
  * **Temps de calcul :** $132.16$ secondes ($\approx 2.20$ minutes)
  * **Taille d'entrée :** $28 \times 28$ (Grayscale)

* **AlexNet (Fine-Tuned) :**
  * **Précision (Test Accuracy) :** $98.85\%$
  * **Min Test Loss :** $0.022289$
  * **Temps de calcul :** $353.20$ secondes ($\approx 5.89$ minutes)
  * **Taille d'entrée :** $224 \times 224$ (RGB)

### Interprétation Métier & Bilan Pratique :
1. **Rendement et pertinence de l'architecture :**  
   Bien que la perte minimale d'AlexNet soit légèrement plus basse ($0.0222$ vs $0.0258$), le modèle **CNN Custom** obtient une précision supérieure sur le jeu de test ($98.97\%$ contre $98.85\%$).
2. **Coût de calcul (Overhead computational) :**  
   AlexNet requiert près de **$2.7$ fois plus de temps GPU** ($353\text{s}$ contre $132\text{s}$). Cet écart s'explique par la nécessité de redimensionner chaque image de $28 \times 28$ vers $224 \times 224$ pixels et de dupliquer les canaux en $3$ dimensions RGB pour satisfaire l'entrée du modèle pré-entraîné sur ImageNet.
3. **Conclusion pour la classification simple :**  
   Pour des tâches simples et à petite échelle comme MNIST, construire une architecture légère *from scratch* est nettement plus efficace et économique qu'adapter une grande architecture complexe conçue pour des images haute résolution.

---

## 2. Partie 2 : Détection d'Objets avec Faster R-CNN

### Analyse des Résultats (Penn-Fudan Pedestrian Dataset)

1. **Convergence Rapide :**
   * **Époque 1 :** Loss = $0.2757$
   * **Époque 3 :** Loss = $0.0992$ (réduction de plus de $64\%$)
   * **Temps d'entraînement :** $106.17$ secondes sur 2x GPU T4.

2. **Performance Visuelle de Localisation et Classification :**
   * **Image Test #1 :** $7$ piétons détectés avec une confiance allant jusqu'à $99.6\%$. Les personnes en arrière-plan et partiellement masquées sont correctement délimitées.
   * **Image Test #2 :** $3$ piétons isolés repérés sans aucun faux positif avec des scores de confiance de $99.2\%$ à $99.6\%$.
   * **Image Test #3 :** $8$ piétons détectés dans une scène dense de passage piéton. Le modèle gère efficacement le chevauchement partiel (*occlusion*).

---

## 3. Synthèse : Ce que nous avons appris (En bref)

* **Adaptation du modèle aux données :** Un modèle lourd (comme AlexNet) n'est pas systématiquement meilleur qu'un petit CNN bien dimensionné si la donnée source (images $28 \times 28$) est simple.
* **Classification vs Détection :** 
  * La **classification** attribue une étiquette unique à l'ensemble d'une image.
  * La **détection** (Faster R-CNN) prédit à la fois **des boîtes englobantes (localisation par régression)** et **des classes d'objets (classification)** pour plusieurs éléments distincts dans une même scène.
* **Intérêt du Transfer Learning :** Pour des tâches complexes comme la détection d'objets sur des images réelles, l'utilisation d'un squelette pré-entraîné (*ResNet-50-FPN*) permet d'obtenir un modèle performant en moins de $2$ minutes d'entraînement.
* **Parallélisation GPU :** L'utilisation de PyTorch avec `nn.DataParallel` sur multi-GPU T4 accélère considérablement le temps d'exécution des pipelines de Deep Learning.
```eof

Ce document résume l'intégralité des analyses et des conclusions de votre travail. N'hésitez pas si vous souhaitez modifier un point particulier pour votre rapport final !