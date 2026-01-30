import numpy as np

class MultiClassAdaline:
    """
    Classificateur Adaline (Adaptive Linear Neuron) Multi-classes.
    
    Paramètres
    ------------
    eta : float
        Taux d'apprentissage (entre 0.0 et 1.0)
    n_iter : int
        Nombre d'époques d'entraînement.
    random_state : int
        Graine pour la génération de nombres aléatoires (reproductibilité).
        
    Attributs
    -----------
    w_ : array-like, shape = [n_features, n_classes]
        Poids synaptiques.
    b_ : array-like, shape = [n_classes]
        Unités de biais (un par classe).
    cost_ : list
        Valeur de la fonction de coût (Somme des erreurs quadratiques) à chaque époque.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Entraînement selon la règle d'apprentissage Adaline (Gradient Descent).
        Voir Diapositive 23 du cours Leçon #2.
        """
        rgen = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialisation (Diapo 9: poids à 0 ou petit nombre aléatoire)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        self.b_ = np.zeros(n_classes) # Le biais (w0 dans le cours)
        
        # Encodage One-Hot pour le One-vs-All (Diapo 13)
        y_enc = np.zeros((X.shape[0], n_classes))
        for idx, cls in enumerate(self.classes_):
            y_enc[:, idx] = np.where(y == cls, 1, -1)

        self.cost_ = []

        for _ in range(self.n_iter):
            # 1. Entrée nette (Diapo 4: z = w^T * x)
            net_input = self.net_input(X)
            
            # 2. Activation linéaire (Diapo 16: phi(z) = z)
            output = net_input
            
            # 3. Calcul de l'erreur (y - output)
            # Note: Le cours utilise (y - output) à la diapo 23 pour le gradient
            errors = y_enc - output
            
            # 4. Mise à jour des poids (Diapo 23 et 24)
            # Formule: w = w + eta * X.T.dot(errors)
            self.w_ += self.eta * X.T.dot(errors)
            
            # Mise à jour du biais (w0) : Somme des erreurs * eta
            self.b_ += self.eta * errors.sum(axis=0)
            
            # 5. Fonction de coût SSE (Diapo 17: J(w) = 1/2 * sum(errors^2))
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
        return self

    def net_input(self, X):
        """Calcule l'entrée nette z = Wx + b"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Retourne l'indice de la classe avec la plus forte activation"""
        # Argmax sur l'axe des colonnes (classes)
        return self.classes_[np.argmax(self.net_input(X), axis=1)]