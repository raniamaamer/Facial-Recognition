import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib

# -------------------------------
# PARAMÈTRES
# -------------------------------
IMAGE_SIZE = (100, 100)
DATA_DIR = "originalimages_part2"

# Vérifier si le dossier existe
if not os.path.exists(DATA_DIR):
    st.error(f"❌ Le dossier '{DATA_DIR}' n'existe pas!")
    st.info(f"📁 Veuillez créer le dossier '{DATA_DIR}' et y mettre vos images")
    st.stop()

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    st.error(f"❌ Erreur chargement Haar Cascade: {e}")
    st.stop()

def extraire_visage(img):
    """Extrait et redimensionne le visage d'une image"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        return face
    except Exception as e:
        st.error(f"Erreur extraction visage: {e}")
        return None

# -------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------
@st.cache_data
def charger_donnees(path):
    """Charge les données avec cache pour performances"""
    images, labels = [], []
    
    if not os.path.exists(path):
        return np.array([]), np.array([])
        
    files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if len(files) == 0:
        st.error("❌ Aucune image trouvée dans le dossier!")
        return np.array([]), np.array([])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        try:
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            face = extraire_visage(img)
            if face is None:
                continue

            flat = face.flatten()
            
            # Extraction de l'ID personne
            if '-' in file:
                person_id = int(file.split("-")[0])
            else:
                # Alternative pour autres formats de nommage
                person_id = int(''.join(filter(str.isdigit, file)) or 1)
            
            images.append(flat)
            labels.append(person_id - 1)
            
            # Mise à jour progression
            if i % 10 == 0:
                progress = (i + 1) / len(files)
                progress_bar.progress(progress)
                status_text.text(f"📁 Chargement... {i+1}/{len(files)} images")
                
        except Exception as e:
            st.warning(f"⚠️ Erreur avec {file}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return np.array(images), np.array(labels)

# -------------------------------
# INTERFACE STREAMLIT
# -------------------------------
st.set_page_config(page_title="Reconnaissance Faciale PCA + LDA + ML", layout="wide")
st.title("🧠 Reconnaissance Faciale : PCA + LDA + Machine Learning")

# Sidebar pour paramètres
st.sidebar.header("⚙️ Paramètres du modèle")
model_choice = st.sidebar.selectbox(
    "Choix du modèle",
    ["Random Forest", "SVM", "MLP (Neural Network)"],
    index=0
)

# Paramètres selon le modèle
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Nombre d'arbres", 50, 300, 100)
    max_depth = st.sidebar.slider("Profondeur max", 5, 50, 20)
elif model_choice == "SVM":
    kernel_type = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    C_value = st.sidebar.slider("Paramètre C", 0.1, 10.0, 1.0)
else:  # MLP
    hidden_layers = st.sidebar.slider("Neurones cachés", 50, 200, 100)
    learning_rate = st.sidebar.selectbox("Taux d'apprentissage", [0.001, 0.01, 0.1])

# -------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------
with st.spinner("📁 Chargement des données..."):
    X, y = charger_donnees(DATA_DIR)

if len(X) == 0:
    st.error("❌ Aucune donnée chargée! Vérifiez le dossier de données.")
    st.stop()

st.success(f"✅ {len(X)} images chargées – {len(np.unique(y))} classes détectées")

# Affichage des statistiques
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Images totales", len(X))
with col2:
    st.metric("Nombre de classes", len(np.unique(y)))
with col3:
    st.metric("Dimensions par image", f"{X.shape[1]} features")

# -------------------------------
# PRÉPARATION DES DONNÉES
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
n_components_pca = min(100, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# LDA
n_components_lda = min(len(np.unique(y)) - 1, 30, X_pca.shape[0], X_pca.shape[1])
if n_components_lda > 0:
    lda = LDA(n_components=n_components_lda)
    X_lda = lda.fit_transform(X_pca, y)
else:
    st.error("❌ Pas assez de classes pour LDA!")
    st.stop()

# Split données
X_train, X_test, y_train, y_test = train_test_split(
    X_lda, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# ENTRAÎNEMENT DU MODÈLE
# -------------------------------
st.header("🎯 Entraînement du modèle")

if model_choice == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model_name = "Random Forest"
    
elif model_choice == "SVM":
    model = SVC(
        kernel=kernel_type,
        C=C_value,
        probability=True,
        random_state=42
    )
    model_name = "SVM"
    
else:  # MLP
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layers,),
        learning_rate_init=learning_rate,
        random_state=42,
        max_iter=500
    )
    model_name = "MLP Neural Network"

with st.spinner(f"⏳ Entraînement du {model_name}..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

st.success(f"✅ {model_name} entraîné avec succès!")

# -------------------------------
# ÉVALUATION
# -------------------------------
st.header("📊 Évaluation des performances")

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

# Métriques en colonnes
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exactitude", f"{accuracy:.2f}%")
with col2:
    st.metric("Précision", f"{precision:.2f}%")
with col3:
    st.metric("Rappel", f"{recall:.2f}%")
with col4:
    st.metric("F1-Score", f"{f1:.2f}%")

# -------------------------------
# MATRICE DE CONFUSION
# -------------------------------
st.subheader("🎯 Matrice de confusion")
fig_cm, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Prédit')
ax.set_ylabel('Réel')
ax.set_title(f'Matrice de Confusion - {model_name}')
st.pyplot(fig_cm)

# -------------------------------
# VISUALISATIONS
# -------------------------------
st.header("📈 Visualisations")

tab1, tab2, tab3 = st.tabs(["PCA vs LDA", "Variance PCA", "Features importantes"])

with tab1:
    fig_vis, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
    ax1.set_title("Projection PCA (2D)")
    ax1.set_xlabel("Composante 1")
    ax1.set_ylabel("Composante 2")
    plt.colorbar(scatter1, ax=ax1)
    
    scatter2 = ax2.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="viridis", alpha=0.6)
    ax2.set_title("Projection LDA (2D)")
    ax2.set_xlabel("Composante 1")
    ax2.set_ylabel("Composante 2")
    plt.colorbar(scatter2, ax=ax2)
    
    st.pyplot(fig_vis)

with tab2:
    fig_var, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
    ax.set_title("Variance cumulée expliquée par PCA")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance cumulée")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_var)

with tab3:
    if model_choice == "Random Forest":
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        ax.bar(range(len(indices)), importances[indices])
        ax.set_title("Top 20 des features les plus importantes (Random Forest)")
        ax.set_xlabel("Index de la feature")
        ax.set_ylabel("Importance")
        st.pyplot(fig_imp)
    else:
        st.info("📊 L'analyse des features importantes est disponible pour Random Forest")

# -------------------------------
# PRÉDICTION EN TEMPS RÉEL
# -------------------------------
st.header("🎯 Prédiction sur nouvelles images")

uploaded_files = st.file_uploader(
    "Téléchargez des images pour tester le modèle",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="predictor"
)

def predict_image(img):
    """Prédit la classe d'une image"""
    face = extraire_visage(img)
    if face is None:
        return None, None, None
    
    flat = face.flatten()
    scaled = scaler.transform([flat])
    pca_feat = pca.transform(scaled)
    lda_feat = lda.transform(pca_feat)
    
    pred_class = model.predict(lda_feat)[0]
    
    if hasattr(model, 'predict_proba'):
        confidence = np.max(model.predict_proba(lda_feat)) * 100
    else:
        confidence = 100.0  # Si pas de probabilités, on met 100%
    
    return pred_class, confidence, face

def trouver_images_similaires(person_id, max_images=6):
    """Trouve des images similaires de la même personne"""
    similar_images = []
    for file in os.listdir(DATA_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                # Extraction ID depuis le nom de fichier
                if '-' in file:
                    file_person_id = int(file.split("-")[0])
                else:
                    file_person_id = int(''.join(filter(str.isdigit, file)) or 1)
                
                if file_person_id == person_id + 1:  # +1 car nos labels commencent à 0
                    img_path = os.path.join(DATA_DIR, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        similar_images.append(img_rgb)
                        if len(similar_images) >= max_images:
                            break
            except:
                continue
    return similar_images

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Affichage image originale
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Image originale", use_column_width=True)
            
            # Réinitialiser le pointeur du fichier pour la prédiction
            uploaded_file.seek(0)
        
        with col2:
            # Prédiction
            pred_class, confidence, face = predict_image(img)
            
            if pred_class is not None:
                st.success(f"✅ **Personne identifiée : {pred_class + 1}**")
                st.metric("Confiance", f"{confidence:.2f}%")
                
                if face is not None:
                    st.image(face, caption="Visage détecté et prétraité", width=200)
                
                # Affichage images similaires
                st.subheader("🔍 Images similaires de cette personne")
                similar_images = trouver_images_similaires(pred_class)
                
                if similar_images:
                    cols = st.columns(3)
                    for idx, sim_img in enumerate(similar_images[:6]):
                        with cols[idx % 3]:
                            st.image(sim_img, width=120)
                else:
                    st.info("Aucune autre image trouvée pour cette personne")
                    
            else:
                st.error("❌ Aucun visage détecté dans l'image")

# -------------------------------
# SAUVEGARDE DU MODÈLE
# -------------------------------
st.sidebar.header("💾 Sauvegarde")

if st.sidebar.button("💾 Sauvegarder le modèle"):
    try:
        model_data = {
            'model': model,
            'scaler': scaler,
            'pca': pca,
            'lda': lda,
            'image_size': IMAGE_SIZE
        }
        joblib.dump(model_data, 'modele_reconnaissance_faciale.pkl')
        st.sidebar.success("✅ Modèle sauvegardé avec succès!")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur sauvegarde: {e}")

# -------------------------------
# INFORMATIONS SYSTÈME
# -------------------------------
with st.sidebar:
    st.header("ℹ️ Informations")
    st.write(f"📊 Images: {len(X)}")
    st.write(f"🎯 Classes: {len(np.unique(y))}")
    st.write(f"🔢 Features: {X.shape[1]}")
    st.write(f"🧠 Modèle: {model_name}")
    st.write("✅ **TensorFlow non requis**")

# Footer
st.markdown("---")
st.markdown("*Système de reconnaissance faciale utilisant PCA + LDA + Machine Learning*")