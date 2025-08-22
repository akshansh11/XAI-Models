import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Install required packages (uncomment if needed):
# !pip install lime shap plotly

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("LIME not available. Install with: pip install lime")
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

class XAIModelAnalyzer:
    def __init__(self):
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
        # Set up beautiful color palettes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#6A994E',
            'warning': '#F4A261',
            'gradient': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        }
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, dataset='wine'):
        """Load and prepare dataset"""
        if dataset == 'wine':
            data = load_wine()
            self.feature_names = data.feature_names
            self.class_names = data.target_names
        elif dataset == 'breast_cancer':
            data = load_breast_cancer()
            self.feature_names = data.feature_names
            self.class_names = data.target_names
        else:
            # Create synthetic dataset
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                                     n_redundant=2, n_clusters_per_class=1, random_state=42)
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.class_names = ['Class_0', 'Class_1']
            data = type('Dataset', (), {'data': X, 'target': y})()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Dataset loaded: {dataset}")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Classes: {len(self.class_names)}")
    
    def train_models(self):
        """Train multiple models"""
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        for name, model in models_to_train.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            print(f"{name}: Accuracy = {accuracy:.3f}")
    
    def plot_accuracy_comparison(self):
        """Plot model accuracy comparison"""
        plt.figure(figsize=(10, 6))
        models = list(self.models.keys())
        accuracies = [self.models[model]['accuracy'] for model in models]
        
        bars = plt.bar(models, accuracies, color=self.colors['gradient'][:len(models)])
        plt.title('Model Accuracy Comparison', fontsize=16)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name='Random Forest'):
        """Plot feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        model = self.models[model_name]['model']
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importance")
            return
        
        plt.figure(figsize=(12, 8))
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]
        
        plt.barh(range(len(indices)), importance[indices], 
                color=sns.color_palette("viridis", len(indices)))
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.title(f'{model_name} Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name='Random Forest'):
        """Plot confusion matrix for a model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.models[model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_distribution(self):
        """Plot distribution of predictions across models"""
        plt.figure(figsize=(10, 6))
        
        for i, (model_name, model_info) in enumerate(self.models.items()):
            preds = model_info['predictions']
            plt.hist(preds, alpha=0.7, label=model_name, bins=len(self.class_names),
                    color=self.colors['gradient'][i % len(self.colors['gradient'])])
        
        plt.title('Prediction Distribution Across Models', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def explain_with_lime(self, model_name='Random Forest', instance_idx=0):
        """LIME explanations with visualizations"""
        if not LIME_AVAILABLE:
            print("LIME not available. Please install with: pip install lime")
            return
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        model = self.models[model_name]['model']
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        # Get explanation for a specific instance
        exp = explainer.explain_instance(
            self.X_test[instance_idx], 
            model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Create LIME visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # LIME feature importance
        lime_values = []
        lime_features = []
        for feature, importance in exp.as_list():
            lime_features.append(feature)
            lime_values.append(importance)
        
        colors = ['red' if x < 0 else 'green' for x in lime_values]
        ax1.barh(range(len(lime_values)), lime_values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(lime_values)))
        ax1.set_yticklabels(lime_features, fontsize=10)
        ax1.set_xlabel('LIME Importance', fontsize=12)
        ax1.set_title(f'LIME Explanation for {model_name} - Instance {instance_idx}', fontsize=14)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.invert_yaxis()
        
        # Feature values for the explained instance
        feature_vals = self.X_test[instance_idx]
        colors_vals = sns.color_palette("coolwarm", len(feature_vals))
        ax2.bar(range(len(feature_vals)), feature_vals, color=colors_vals, alpha=0.8)
        ax2.set_xticks(range(len(feature_vals)))
        ax2.set_xticklabels([f'F{i}' for i in range(len(feature_vals))], rotation=45)
        ax2.set_ylabel('Feature Value', fontsize=12)
        ax2.set_title('Feature Values for Explained Instance', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def explain_with_shap(self, model_name='Random Forest', n_samples=50):
        """SHAP explanations with visualizations"""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Please install with: pip install shap")
            return
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        model = self.models[model_name]['model']
        
        try:
            # Create SHAP explainer based on model type
            if 'Random Forest' in model_name or 'Gradient' in model_name:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_test[:n_samples])
            else:
                explainer = shap.Explainer(model, self.X_train[:100])
                shap_values = explainer(self.X_test[:n_samples])
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
            
            # Handle multi-class case - take the first class for visualization
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class for binary classification
            elif len(shap_values.shape) == 3:  # Multi-class case
                shap_values = shap_values[:, :, 1]  # Use positive class
            
            self.plot_shap_summary(shap_values, model_name)
            self.plot_shap_waterfall(shap_values, explainer, model_name, instance_idx=0)
            
        except Exception as e:
            print(f"Error in SHAP analysis: {str(e)}")
            print("This might be due to model compatibility. Trying alternative approach...")
    
    def plot_shap_summary(self, shap_values, model_name):
        """Plot SHAP summary"""
        plt.figure(figsize=(12, 8))
        
        # SHAP feature importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1][:15]
        
        plt.barh(range(len(sorted_idx)), mean_shap[sorted_idx], 
                color=sns.color_palette("viridis", len(sorted_idx)))
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, shap_values, explainer, model_name, instance_idx=0):
        """Plot SHAP waterfall for single instance"""
        plt.figure(figsize=(12, 8))
        
        shap_instance = shap_values[instance_idx]
        
        # Get base value
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0
        
        # Sort features by SHAP value magnitude
        sorted_features = sorted(enumerate(shap_instance), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        colors_waterfall = ['red' if val < 0 else 'green' for _, val in sorted_features]
        values = [val for _, val in sorted_features]
        feature_names = [self.feature_names[idx] for idx, _ in sorted_features]
        
        plt.barh(range(len(sorted_features)), [abs(v) for v in values], 
                color=colors_waterfall, alpha=0.7)
        plt.yticks(range(len(sorted_features)), feature_names)
        plt.xlabel('|SHAP value|', fontsize=12)
        plt.title(f'SHAP Values for Single Instance - {model_name}', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def create_interactive_accuracy_plot(self):
        """Create interactive accuracy comparison with Plotly"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Please install with: pip install plotly")
            return
        
        models = list(self.models.keys())
        accuracies = [self.models[model]['accuracy'] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, 
                  marker_color=self.colors['gradient'][:len(models)],
                  text=[f'{acc:.3f}' for acc in accuracies],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title='Interactive Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Accuracy',
            template='plotly_white',
            font=dict(size=12)
        )
        
        fig.show()
    
    def create_interactive_feature_radar(self, model_name='Random Forest'):
        """Create interactive radar chart for feature importance"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Please install with: pip install plotly")
            return
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        model = self.models[model_name]['model']
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importance")
            return
        
        importance = model.feature_importances_
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=importance,
            theta=self.feature_names,
            fill='toself',
            name='Feature Importance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(importance)]
                )),
            showlegend=True,
            title=f'Feature Importance Radar Chart - {model_name}'
        )
        
        fig.show()
    
    def run_all_plots(self, dataset='wine'):
        """Run all visualizations separately"""
        print("=" * 60)
        print("EXPLAINABLE AI MODEL ANALYSIS")
        print("=" * 60)
        
        # Load data and train models
        self.load_data(dataset)
        self.train_models()
        
        print("\n1. Model Accuracy Comparison")
        self.plot_accuracy_comparison()
        
        print("\n2. Feature Importance Analysis")
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in self.models:
                self.plot_feature_importance(model_name)
        
        print("\n3. Confusion Matrices")
        for model_name in self.models.keys():
            self.plot_confusion_matrix(model_name)
        
        print("\n4. Prediction Distribution")
        self.plot_prediction_distribution()
        
        if LIME_AVAILABLE:
            print("\n5. LIME Explanations")
            self.explain_with_lime('Random Forest', instance_idx=0)
        
        if SHAP_AVAILABLE:
            print("\n6. SHAP Explanations")
            self.explain_with_shap('Random Forest', n_samples=30)
        
        if PLOTLY_AVAILABLE:
            print("\n7. Interactive Plots")
            self.create_interactive_accuracy_plot()
            self.create_interactive_feature_radar('Random Forest')
        
        print("\nAnalysis complete!")


if __name__ == "__main__":
    # Create analyzer instance
    analyzer = XAIModelAnalyzer()
    
    # Run all plots separately
    analyzer.run_all_plots(dataset='wine')  # Try 'breast_cancer' or 'synthetic'
    
    # You can also run individual plots:
    # analyzer.load_data('wine')
    # analyzer.train_models()
    # analyzer.plot_accuracy_comparison()
    # analyzer.plot_feature_importance('Random Forest')
    # analyzer.explain_with_lime('Random Forest', instance_idx=5)
    # analyzer.explain_with_shap('Gradient Boosting', n_samples=50)
