import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from sklearn.neighbors import NearestNeighbors

class PlantAnalysisSystem:
    def __init__(self):
        # Charger les datasets
        try:
            self.dataset = pd.read_csv('plant_dataset.csv')
            if 'Image' not in self.dataset.columns:
                raise ValueError("Column 'Image' not found in plant_dataset.csv")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load plant_dataset.csv: {str(e)}")
        
        self.image_dir = 'media/dataset_images/images'
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Initialiser les modèles
        self.similarity_model = self.init_similarity_model()
        self.disease_model = self.init_disease_model()
        
        # Charger/créer l'index des caractéristiques
        self.knn, self.features = self.load_feature_index()
    
    def init_similarity_model(self):
        """Initialiser le modèle pour la recherche d'images similaires"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    def init_disease_model(self):
        """Initialiser le modèle de détection de maladies"""
        try:
            return load_model('disease_model.h5')
        except:
            print("⚠️ Modèle de maladie non trouvé. Utilisation du mode diagnostic basique.")
            return None
    
    def load_feature_index(self):
        """Charger ou créer l'index des caractéristiques"""
        if os.path.exists('features_index.joblib'):
            try:
                return joblib.load('features_index.joblib')
            except:
                print("⚠️ Failed to load features_index.joblib. Creating new index...")
        print("Création de l'index des caractéristiques...")
        return self.create_feature_index()
    
    def create_feature_index(self):
        """Créer l'index des caractéristiques"""
        features = []
        valid_images = []
        for img_name in self.dataset['Image']:
            img_path = os.path.normpath(os.path.join(self.image_dir, img_name))
            if os.path.exists(img_path):
                try:
                    feature = self.extract_features(img_path)
                    if feature is not None:
                        features.append(feature)
                        valid_images.append(img_name)
                except Exception as e:
                    print(f"⚠️ Failed to process image {img_path}: {str(e)}")
            else:
                print(f"⚠️ Image not found: {img_path}")
        
        if not features:
            print("⚠️ No valid images found for feature extraction. Using empty index.")
            return None, np.array([])
        
        features = np.array(features)
        knn = NearestNeighbors(n_neighbors=3, metric='cosine')
        knn.fit(features)
        joblib.dump((knn, features), 'features_index.joblib')
        print(f"✅ Created feature index with {len(features)} images.")
        return knn, features
    
    def extract_features(self, img_path):
        """Extraire les caractéristiques d'une image"""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            return self.similarity_model.predict(x, verbose=0).flatten()
        except Exception as e:
            print(f"⚠️ Error extracting features from {img_path}: {str(e)}")
            return None
    
    def detect_disease(self, img_path):
        """Détecter les maladies dans une image"""
        if not self.disease_model:
            return 'sain', 1.0
            
        try:
            img = image.load_img(img_path, target_size=(256, 256))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            predictions = self.disease_model.predict(x, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]
            class_names = ['sain', 'mildiou', 'oïdium', 'pourriture', 'rouille']
            return class_names[class_idx], float(confidence)
        except Exception as e:
            print(f"⚠️ Error detecting disease in {img_path}: {str(e)}")
            return 'sain', 1.0
    
    def analyze_similarity(self, img_path):
        """Trouver les images similaires et leurs données"""
        if self.knn is None or len(self.features) == 0:
            print("⚠️ No feature index available. Returning default values.")
            return 50.0, 50.0  # Default values for moisture and waterflow
        
        new_features = self.extract_features(img_path)
        if new_features is None:
            print("⚠️ Failed to extract features for similarity analysis. Returning default values.")
            return 50.0, 50.0
        
        distances, indices = self.knn.kneighbors([new_features])
        similar_data = self.dataset.iloc[indices[0]]
        avg_moisture = similar_data['soil_moisture_subse(%)'].mean()
        avg_waterflow = similar_data['water_flow_subset(%)'].mean()
        return avg_moisture, avg_waterflow
    
    def generate_advice(self, moisture, waterflow, disease_status):
        """Générer des conseils basés sur les analyses"""
        advice = []
        
        if moisture < 30:
            advice.append("🔴 Le sol est trop sec. Augmentez l'arrosage.")
        elif moisture > 70:
            advice.append("🔴 Attention! Sol trop humide. Réduisez l'arrosage.")
        else:
            advice.append("🟢 Humidité du sol optimale.")
        
        if waterflow < 10:
            advice.append("🔴 Débit d'eau trop faible. Vérifiez votre système d'irrigation.")
        elif waterflow > 50:
            advice.append("🔴 Débit d'eau trop élevé. Vous gaspillez de l'eau.")
        else:
            advice.append("🟢 Débit d'eau approprié.")
        
        if disease_status != 'sain':
            advice.append(f"⚠️ Maladie détectée: {disease_status}. Traitez avec un fongicide approprié.")
        else:
            advice.append("🌿 Votre plante est en bonne santé.")
        
        if moisture < 30 and disease_status != 'sain':
            advice.append("💧 Augmentez l'arrosage mais évitez de mouiller les feuilles pour limiter la propagation.")
        
        return advice
    
    def analyze_plant(self, img_path):
        """Analyser complètement une plante"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Fichier introuvable: {img_path}")
        
        moisture, waterflow = self.analyze_similarity(img_path)
        disease_status, confidence = self.detect_disease(img_path)
        advice = self.generate_advice(moisture, waterflow, disease_status)
        return moisture, waterflow, disease_status, confidence, advice

def main():
    # Initialiser le système
    print("⏳ Initialisation du système d'analyse...")
    try:
        system = PlantAnalysisSystem()
        print("✅ Système prêt!")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {str(e)}")
        return
    
    # Demander le chemin de l'image
    img_path = input("\nEntrez le chemin complet de l'image à analyser: ")
    
    try:
        # Effectuer l'analyse
        print("\n🔍 Analyse en cours...")
        moisture, waterflow, disease, confidence, advice = system.analyze_plant(img_path)
        
        # Afficher les résultats
        print("\n📊 RESULTATS DE L'ANALYSE")
        print("--------------------------------")
        print(f"💧 Humidité du sol: {moisture:.1f}%")
        print(f"🚰 Débit d'eau: {waterflow:.1f}%")
        print(f"🌱 État de santé: {disease} (confiance: {confidence*100:.1f}%)")
        
        print("\n💡 CONSEILS:")
        for tip in advice:
            print(f" - {tip}")
            
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")

if __name__ == "__main__":
    main()