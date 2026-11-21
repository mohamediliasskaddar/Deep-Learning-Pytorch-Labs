La supériorité des architectures à portes (LSTM et GRU) : Les modèles LSTM et GRU obtiennent une accuracy de 46,4 %, soit presque 3 fois mieux que le RNN simple (17,9 %) et près de 2 fois mieux que le BiRNN (25,0 %). Comme tes articles en arabe peuvent atteindre 400 tokens (max_len), le RNN et le BiRNN souffrent de la disparition du gradient (vanishing gradient) et oublient le début du texte, tandis que les portes de mémoire du LSTM et du GRU conservent l'information pertinente tout au long de l'article.

LSTM vs GRU (Performance vs Vitesse) : Le LSTM obtient le meilleur F1-score macro (0,456 contre 0,410 pour le GRU) grâce à une meilleure précision globale (0,472 vs 0,426). En revanche, le GRU est presque 3 fois plus rapide à entraîner (0,67 s contre 1,76 s) car il possède une seule couche (layers=1, hidden=64) contre deux couches (layers=2, hidden=128) pour le LSTM.

Voici une analyse complète et structurée de tes résultats, que tu peux utiliser directement pour comprendre tes modèles et rédiger ton compte-rendu.

---

### 1. Tableau comparatif global : le **LSTM** est le meilleur modèle

Le tableau final montre deux groupes très nets :

| Modèle | Accuracy | Precision (macro) | Recall (macro) | **F1 (macro)** | Temps (s) |
| --- | --- | --- | --- | --- | --- |
| **LSTM** | **46,4 %** | **0,472** | **0,453** | **0,456** | 1,76 s |
| **GRU** | **46,4 %** | 0,426 | 0,446 | 0,410 | **0,67 s** |
| **BiRNN** | 25,0 % | 0,236 | 0,242 | 0,238 | 2,74 s |
| **RNN** | 17,9 % | 0,175 | 0,183 | 0,179 | 1,53 s |

* **La supériorité des architectures à portes (LSTM et GRU) :** Les modèles **LSTM** et **GRU** obtiennent une accuracy de **46,4 %**, soit presque **3 fois mieux que le RNN simple** (17,9 %) et près de **2 fois mieux que le BiRNN** (25,0 %). Comme tes articles en arabe peuvent atteindre 400 tokens (`max_len`), le RNN et le BiRNN souffrent de la disparition du gradient (*vanishing gradient*) et oublient le début du texte, tandis que les portes de mémoire du LSTM et du GRU conservent l'information pertinente tout au long de l'article.
* **LSTM vs GRU (Performance vs Vitesse) :** Le **LSTM** obtient le meilleur **F1-score macro (0,456 contre 0,410 pour le GRU)** grâce à une meilleure précision globale (0,472 vs 0,426). En revanche, le **GRU** est presque **3 fois plus rapide** à entraîner (0,67 s contre 1,76 s) car il possède une seule couche (`layers=1`, `hidden=64`) contre deux couches (`layers=2`, `hidden=128`) pour le LSTM.

---

### 2. Analyse par classe (Matrices de confusion et `classification_report`)

Tes classes semblent correspondre à des niveaux de pertinence footballistique dérivés de l'histogramme (`Score` de 0 à 10) : **Classe 0** (faible/non pertinent), **Classe 1** (pertinence moyenne/intermédiaire), **Classe 2** (forte pertinence).

* **Les classes extrêmes (0 et 2) sont bien détectées par le LSTM et le GRU :**
* **LSTM :** Il atteint un **F1-score de 0,57 sur la classe 0** (précision de 0,67, 0 confusion totale entre classe 2 et classe 0) et **0,56 sur la classe 2** (7 vrais positifs sur 11).
* **GRU :** Il excelle particulièrement sur la **classe 2** avec un **F1-score de 0,59** et un rappel de **73 %** (8 textes bien classés sur 11), mais au prix de plus de faux positifs sur la classe 1.


* **La classe 1 est le point faible de tous les modèles :**
* Pour le LSTM (F1 = **0,24**) comme pour le GRU (F1 = **0,17**), la classe 1 est majoritairement confondue avec la classe 2 (5 articles de la classe 1 sont prédits en classe 2 dans les deux matrices).
* **Explication par tes données (l'histogramme) :** L'histogramme du *Football relevance score* montre une distribution très concentrée entre `0` et `3` (un grand pic), un petit rebond vers `10` (les articles 100 % football), et très peu d'exemples dans la zone intermédiaire (`3` à `8`). La frontière sémantique entre un article "moyennement lié au football" (classe 1) et "fortement lié" (classe 2) est donc floue, d'autant plus que ton jeu de test ne contient que **28 exemples au total** (`support`: 8, 9 et 11).



---

### 3. Interprétation des hyperparamètres choisis par Optuna (`best_cfg`)

* **Taille du dataset (`support = 28` en test) :** Avec un jeu de test de 28 articles (15 % du total), ton dataset complet contient environ **185 à 190 articles** (soit ~130 articles en `train`). C'est un jeu de données de petite taille pour du *Deep Learning* entraîné de zéro (*from scratch*).
* **Choix de régularisation :** Optuna a systématiquement retenu un **dropout élevé (`0.5` pour GRU et LSTM, `0.3` pour RNN et BiRNN)** afin d'éviter le surapprentissage sur ce petit volume de données.
* **Le BiRNN est trop lourd pour ce dataset :** Optuna lui a attribué la plus grande capacité (`emb=300`, `hidden=256`, `layers=2`) et un taux d'apprentissage très faible (`lr=0.00025`). Résultat : c'est le modèle le plus lent (2,74 s) et il n'arrive pas à converger efficacement sur 130 exemples d'entraînement (F1 = 0,238).

---

### 4. Pistes d'amélioration concrètes (à mentionner en conclusion de ton rapport)

1. **Utiliser des embeddings pré-entraînés (AraVec ou FastText arabe) :** Sur ~130 textes d'entraînement, entraîner une couche `nn.Embedding` de taille 300 de zéro limite la compréhension du vocabulaire. Initialiser les embeddings avec des vecteurs pré-entraînés améliorerait fortement le F1-score.
2. **Utiliser un modèle Transformer arabe (AraBERT / MARBERT) :** Sur de petits jeux de données textuels, le *fine-tuning* d'un modèle pré-entraîné comme AraBERT dépasse généralement les architectures RNN/LSTM entraînées *from scratch*.