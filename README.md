# 🏫 ClassroomAI — Inspection intelligente de salles de classe

Application d’analyse automatisée de salles de classe basée sur l’IA (Computer Vision + NLP), permettant de détecter des anomalies, vérifier la conformité et générer des dashboards analytiques en temps réel.

## 🚀 Accès à l’application

👉 **Lien direct :** https://dkpn7gdbaabpieksxybh6w.streamlit.app/

---

## 🧠 Fonctionnalités principales

### 🔍 Analyse intelligente d’images

* Détection d’objets avec **YOLOv8**
* Génération de description avec **BLIP**
* Extraction de texte via **OCR (Tesseract)**

### ⚠️ Détection d’anomalies

* Effectif incorrect (nombre d’élèves)
* TV éteinte ou absente
* Fenêtre ouverte
* Déchets au sol
* Objets interdits (téléphone, bouteille, etc.)
* Chaise renversée
* Personne debout
* Cartable mal placé ou abandonné

### 📊 Dashboard analytique

* Nombre de salles inspectées
* Taux de conformité
* Visualisation des anomalies
* Répartition des objets détectés

### 🗄️ Gestion des données

* Base de données interactive
* Filtres dynamiques
* Visualisation détaillée

### 📥 Export des résultats

* Export CSV global
* Export ZIP (par salle)

---

## 🛠️ Stack technique

* **Frontend / App** : Streamlit
* **Computer Vision** : OpenCV, YOLOv8 (Ultralytics)
* **IA Vision-Langage** : BLIP (Salesforce)
* **OCR** : Tesseract
* **Data** : Pandas, NumPy
* **Visualisation** : Plotly

---

## 📦 Installation (local)

```bash
git clone https://github.com/ton-repo/classroom-ai.git
cd classroom-ai

pip install -r requirements.txt

streamlit run app.py
```

---

## 📁 Structure du projet

```
📦 classroom-ai
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📂 assets/
 ┗ 📜 README.md
```

---

## ⚙️ Règles métier implémentées

L’application intègre des règles spécifiques pour analyser les salles :

* Nombre attendu d’élèves : **9**
* Détection de posture (debout vs assis)
* Analyse de luminosité (qualité image + TV)
* Analyse contextuelle via caption IA
* Géométrie des objets (position, orientation)

---

## 📸 Utilisation

1. Aller sur la page **Upload**
2. Importer des images de salles
3. Lancer l’analyse
4. Consulter :

   * Dashboard 📊
   * Données détaillées 🗄️
   * Export 📥

---

## 🔐 Limitations

* Dépend de la qualité des images
* OCR variable selon la netteté
* Détection basée sur modèles pré-entraînés (non custom)

---

## 💡 Améliorations possibles

* Entraînement YOLO custom (dataset école)
* Ajout de tracking vidéo
* Détection de comportements (attention, interaction)
* Authentification utilisateur
* Base de données persistante (SQL)

---

## 👨‍💻 Auteur

Projet développé dans le cadre d’un système d’inspection automatisée basé sur l’intelligence artificielle.

---

## ⭐️ Support

Si le projet t’aide, pense à mettre une ⭐ sur GitHub !
