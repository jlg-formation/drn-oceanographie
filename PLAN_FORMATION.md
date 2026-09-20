# Formation : Deep Learning et réseaux de neurones : les fondamentaux

## Introduction

La présentation de l'intelligence artificielle (IA) dans les grands médias relève souvent du fantasme. Ce séminaire vous permettra de clarifier les concepts clés des réseaux de neurones profonds (deep learning) ainsi que leurs différents domaines d'utilisation. Des démonstrations présenteront des cas d'utilisations variés, avec un ancrage concret sur des données et usages de l'océanographie.

## Objectifs pédagogiques

À l’issue de la formation, le participant sera en mesure de :

- Comprendre les concepts de machine learning et l’évolution vers le deep learning (réseaux de neurones profonds)
- Maîtriser les bases théoriques et pratiques d'architecture et de convergence de réseaux de neurones
- Connaître les différentes architectures fondamentales existantes et maîtriser leurs implémentations fondamentales
- Maîtriser les méthodologies de mise en place de réseaux de neurones, les points forts et les limites de ces outils
- Connaître les briques de base du deep learning : réseaux de neurones simples, convolutifs et récursifs
- Appréhender les modèles plus avancés : auto-encodeurs, gans, apprentissage par renforcement

## Public concerné

Toute personne intéressée par le deep learning et les réseaux de neurones : ingénieurs, analystes, data scientists, data analysts, data stewards, développeurs… notamment lorsqu'ils travaillent sur des données scientifiques, géospatiales ou océanographiques.

## Prérequis

Avoir des bases en programmation et une bonne maîtrise des outils informatiques et statistiques. Connaître les bases du machine learning est recommandé.

## Méthodes et moyens pédagogiques

### Méthodes pédagogiques

Ce séminaire se base sur des présentations, des échanges et des études de cas. Des outils comme Lasagne ou Keras seront présentés. Les exemples pratiques s'appuieront autant que possible sur des jeux de données inspirés de l'observation océanique, de la bathymétrie, de la météo marine et de la surveillance du milieu maritime.

## Modalités d'évaluation

Le formateur évalue la progression pédagogique du participant tout au long de la formation au moyen de QCM, mises en situation, travaux pratiques…
Le participant complète également un test de positionnement en amont et en aval pour valider les compétences acquises.

## Programme de la formation

## Chapitre 1 : Introduction IA, machine learning et deep learning

- Historique, concepts de base et applications de l'intelligence artificielle loin des fantasmes portés par ce domaine.
- Intelligence collective : agréger une connaissance partagée par de nombreux agents virtuels.
- Algorithmes génétiques : faire évoluer une population d'agents virtuels par sélection.
- Machine learning usuel : définition.
- Types de tâches : supervised learning, unsupervised learning, reinforcement learning.
- Types d'actions : classification, régression, clustering, estimation de densité, réduction de dimensionalité.
- Exemples d'algorithmes de machine learning : régression linéaire, Naive Bayes, Random Tree.
- Machine learning versus deep learning : pourquoi le ML reste aujourd'hui l'état de l'art (Random Forests & XGBoosts) ?

**Travaux pratiques** : Identifier, sur des cas issus de données marines et environnementales, quels problèmes relèvent de la classification, de la régression, du clustering ou de la prévision.

## Chapitre 2 : Concepts fondamentaux d'un réseau de neurones

- Rappel de bases mathématiques.
- Le réseau de neurones : architecture, fonctions d'activation et de pondération des activations précédentes...
- L'apprentissage d'un réseau de neurones : fonctions de coût, back-propagation, stochastic gradient descent...
- Modélisation d'un réseau de neurones : modélisation des données d'entrée et de sortie selon le type de problème.
- Appréhender une fonction par un réseau de neurones. Appréhender une distribution par un réseau de neurones.
- Data augmentation : comment equilibrer un dataset ?
- Généralisation des résultats d'un réseau de neurones.
- Initialisations et régularisations d'un réseau de neurones : L1/L2 Regularization, Batch Normalization.
- Optimisations et algorithmes de convergence.

**Travaux pratiques** : Approximation d'une fonction et d'une distribution par un réseau de neurones, à partir de séries simplifiées de température, salinité ou hauteur de houle.

## Chapitre 3 : Outils usuels machine learning et deep learning

- Calcul scientifique : NumPy, SciPy.
- Manipulation de données : pandas, Polars.
- Données scientifiques multidimensionnelles : xarray, NetCDF4, HDF5.
- Machine learning : scikit-learn.
- Deep learning : PyTorch, TensorFlow/Keras, JAX.
- Traitement de données volumineuses : Dask, Apache Spark.
- Visualisation : Matplotlib, Plotly.
- Écosystème et modèles pré-entraînés : Hugging Face.

**Travaux pratiques** : Construire et comparer un petit pipeline de préparation et d'analyse de données océanographiques multi-sources : chargement de données NetCDF, manipulation avec xarray et NumPy, préparation des données, analyse avec scikit-learn ou PyTorch et visualisation des résultats.

## Chapitre 4 : Convolutional Neural Networks (CNN)

- Présentation des CNN : principes fondamentaux et applications.
- Fonctionnement fondamental d'un CNN : couche convolutionnelle, utilisation d'un kernel, padding et stride...
- Architectures CNN ayant porté l'état de l'art en classification d'images : LeNet, VGG Networks, Network in Network...
- Utilisation d'un modèle d'attention.
- Application à un cas de figure de classification usuel (texte ou image).
- CNN pour la génération : super-résolution, segmentation pixel à pixel.
- Principales stratégies d'augmentation des feature maps pour la génération d'une image.

**Travaux pratiques** : Étudier, sur des cartes marines, images satellite ou champs de surface, les apports des architectures CNN pour la détection de structures océaniques, la segmentation et l'amélioration de résolution.

## Chapitre 5 : Recurrent Neural Networks (RNN)

- Présentation des RNN : principes fondamentaux et applications.
- Fonctionnement fondamental du RNN : hidden activation, back propagation through time, unfolded version.
- Evolutions vers les GRU (Gated Recurrent Units) et LSTM (Long Short Term Memory).
- Problèmes de convergence et vanising gradient.
- Types d'architectures classiques : prédiction d'une série temporelle, classification...
- Architecture de type RNN Encoder Decoder. Utilisation d'un modèle d'attention.
- Applications NLP : word/character encoding, traduction.
- Applications vidéo : prédiction de la prochaine image générée d'une séquence vidéo.

**Travaux pratiques** : Comparer GRU et LSTM sur une prévision de séries temporelles marines, par exemple l'évolution de courants, de marées ou de paramètres mesurés par bouées.

## Chapitre 6 : Modèles générationnels : VAE et GAN

- Présentation des modèles générationnels Variational AutoEncoder (VAE) et Generative Adversarial Networks (GAN).
- Auto-encoder : réduction de dimensionalité et génération limitée.
- Variational AutoEncoder : modèle générationnel et approximation de la distribution d'une donnée.
- Définition et utilisation de l'espace latent. Reparameterization trick.
- Fondamentaux du Generative Adversarial Networks.
- Convergence d'un GAN et difficultés rencontrées.
- Convergence améliorée : Wasserstein GAN, BeGAN. Earth Moving Distance.
- Applications de génération d'images ou de photographies, génération de texte, super résolution.

**Travaux pratiques** : Explorer la génération ou la reconstruction de cartes océaniques incomplètes, ainsi que l'usage de l'espace latent pour analyser des situations maritimes typiques.

## Chapitre 7 : Deep Reinforcement Learning

- Reinforcement learning.
- Utilisation d'un réseau de neurones pour appréhender la fonction d'état.
- Deep Q learning : experience replay et application au contrôle d'un jeu vidéo.
- Optimisations de la politique d'apprentissage. On-policy et off-policy. Actor critic architecture. A3C.
- Applications : contrôle d'un jeu vidéo simple ou d'un système numérique.

**Travaux pratiques** : Concevoir un agent simple capable d'optimiser une trajectoire ou une décision dans un environnement maritime simulé, sous contraintes d'état, d'actions et de récompense.