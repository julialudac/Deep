LU DAC Julia		ARHILIUC Cristina
LACHAT Laëtitia	CHAUVET Pierrick

Projet de Conception DEEP - Rapport


Objectif	2
Création du réseau de neurones	2
Description du réseau de neurones (NN)	2
Training	2
Testing	2
Amélioration de performances du classifieur	3
Résolution des classes déséquilibrées	3
Data augmentation	3
Battage (=shuffling) du jeu de données	4
Normalisation du jeu de données	4
Configurations : trouver les hyper-paramètres optimaux	4
Le taux d’apprentissage (= learning rate)	4
Extension multi-modale	4
Détection de visages dans une image	5
Problématique	5
Exploitation des sous-détections	6
Des outputs du réseau de neurones aux sous-détections	6
Clustering & Filtrage	6
Sélection de la meilleure sous-détection	7
L’affichage des résultats	8
Amélioration de performances du détecteur	8
Modification de epsilon	8
Impact des réductions successives de l’image par deux	8
Configurations : trouver les hyper-paramètres optimaux	8
Les variables influençant le nombre de détections	8
Résultats obtenus	9
Amélioration Algorithmique	9
Conclusion	10
Voies d’amélioration	10
Manuel de l’utilisateur	10
Objectif
La problématique de ce projet est de construire un système capable de reconnaître les visages apparaissant sur une image. Pour cela, nous allons utiliser une technique d’apprentissage profond (aussi appelée Deep Learning) via des réseaux de neurones de convolution (=CNN).

	Dans un premier temps, un jeu de données d’images en noir et blanc nous a été fourni pour effectuer les étapes “training” et “testing”. En cherchant à améliorer les performances sur ces données (sans atteindre l’extrême c’est-à-dire le sur-apprentissage (=”over-fitting”)), nous pourrons ensuite tenter d’identifier des visages sur des images quelconques (en RGB).

	Le projet est écrit en Python, en utilisant le framework PyTorch. Le code est disponible à l’adresse suivante : https://github.com/julialudac/Deep 
Création du réseau de neurones
Description du réseau de neurones (NN)
Le choix du réseau de neurones s’est porté sur un réseau convolutif : il s’agit donc d’une succession de couches de convolution et de sous-échantillonnage. La back-propagation permet d'apprendre les poids attribués.
Training
La première étape est donc d’entraîner le réseau de neurones à reconnaître des visages, grâce à des images pour lesquelles le label est connu. C’est donc à cette étape que le réseau de neurones va apprendre les poids.
Le jeu de données fourni pour le training est composé d’images en niveau de gris séparées en deux classes : la classe 0 est représentée par 26950 images quelconques (non-visages), la classe 1 est représentée par 64770 visages.
Testing
Maintenant que le réseau de neurones est entraîné sur les données “training”, nous allons tester ses performances sur de nouvelles images.
Dans un premier temps, nous avons utilisé le jeu de données fourni par les enseignants. Il s’agit d’images en niveau de gris dont la répartition est la suivante : 797 visages et 6831 non-visages (voir les résultats obtenus dans la partie Résultats).
Dans un second temps, nous avons testé le réseau sur des images quelconques, en RGB. Le but final est de pointer les visages présents sur l’image. Pour cela, nous avons donc analysé l’image plusieurs fois à travers une fenêtre (cette fenêtre a une taille variable selon les itérations). 
Amélioration de performances du classifieur
Résolution des classes déséquilibrées
Dans le jeu de données du training, on peut remarquer qu’il y a plus de non-visages que de visages, ce qui est un problème pour la performance. En effet, naturellement, le réseau de neurones va avoir tendance à classifier une image comme non-visage.
Pour contrer ce défaut, il y a plusieurs possibilités : 
En appliquant un principe de data augmentation, nous pouvons finir par équilibrer de manière équitable la proportion de non-visages/visages. Le principe de fonctionnement est expliqué dans la partie suivante.
Nous avons également appliqué un reweighting en donnant plus de poids aux images visages qu’aux images non-visages.
Nous pourrions également choisir d’entraîner notre algorithme sur un jeu de données différent (notamment, un jeu de données avec des images en RGB). Cependant, la constitution d’un tel jeu de données était consommatrice en terme de temps, et nous avons choisi de nous concentrer sur d’autres tâches d’amélioration.
Résultat : 
Dans notre cas du “reweighting” c’est-à-dire des modifications des poids sur les datasets, cela nous a permis de passer d’un score inférieur à 20% (entre 9 et 16%) à un score de 34% pour l’accuracy de la classe 1 (c’est-à-dire la reconnaissance de visages) (sur le dataset testing fourni par les professeurs).
Data augmentation
La position du visage d’un même sujet varie toujours légèrement entre différentes photographies. Cela complique donc la tâche de l’apprentissage. Une solution peut donc d’appliquer plusieurs rotations sur une image : ainsi, l’angle d’apprentissage sera différent et le réseau de neurones sera capable de reconnaître davantage des visages dans des positions diverses. Cela réduira ainsi le risque de sur-apprentissage.
De plus, la data augmentation va multiplier la taille du jeu de données en fonction du nombre de rotations appliquées sur chaque image.
Il est à noter que la rotation d’image n’est qu’un exemple de data augmentation : nous aurions également pu appliquer une opération de renverse (=flipping) ou de colorisation. Nous avons testé l’opération de flipping, mais nous n’avons pas sauvegardé cette opération dans la suite de nos programmes.

Résultat : 
Dans notre cas, nous avons analysé de meilleurs résultats lorsque nous appliquons quatre rotations sur une image (une rotation à 90°, une autre à 180° et une autre à 270°). Nous passions d’une accuracy pour la classe 1 de 34% et 69% (sur le dataset testing fourni par les professeurs). 
Cependant, cela a pour conséquence de quadrupler la taille du jeu de données et va donc considérablement ralentir l’étape de training.
De plus, nous avons testé l’opération de flipping, mais nous n’avons pas sauvegardé cette opération dans la suite de nos programmes. En effet, cela dégradait nos résultats : on passait de 69% de réussites à 65% pour l’accuracy de la classe 1 (sur le dataset testing fourni par les professeurs).
Battage (=shuffling) du jeu de données
	Un réseau de neurones apprend plus rapidement si l’échantillon présenté est “inattendu”. Il est donc recommandé d’alterner la classe/type d’image présentée au réseau lors de l’apprentissage. 
	Dans notre cas, nous avons choisi d’appliquer une opération de “shuffling” lors de la constitution du chargeur de données pour les différents mini-batches.

ex : train_loader = torch.utils.data.DataLoader(train_imagenet, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
Normalisation du jeu de données
	Pour améliorer les performances, il est également recommandé de normaliser les valeurs d’entrée. C’est ce que nous avons fait pour tous les jeux de données fournies (training, testing et les images de testing quelconques).
Configurations : trouver les hyper-paramètres optimaux
Le taux d’apprentissage (= learning rate)
	Lors du training de notre réseau de neurones, l’un des paramètres les plus critiques est le learning rate. En effet, un learning rate optimal permettra de diminuer le loss sans pour autant avoir des fluctuations de loss trop importantes.

Résultat : Dans notre cas, nous avons trouvé un learning rate optimal pour la valeur 0.001.
Extension multi-modale
	Notre réseau de neurones est capable de se nourrir d’images en RGB. Cela permet ainsi de tester notre solution sur un plus grand nombre d’images. 

Détection de visages dans une image
Problématique
L’entrée du programme est une image sur laquelle on souhaite détecter des visages. Comme une sliding window est utilisée, on obtiendra un ensemble de petites images (fragments) à chaque réduction de taille. Ainsi, chaque fois que l’on réduit l’image d’un facteur de 1.2 (c’est-à-dire le division_factor), on obtient un dataset contenant les petites images capturées (image chunks or frames) par la sliding window. Il faut aussi sauver les positions des fragments d’images (un fragment d’image est défini par son coin supérieur gauche). On fournit ces frames au réseau de neurones pour faire des prédictions sur la classe de chacune. Les résultats étant très variables, on renouvelle l’expérience 3 fois sur la même image pour ne garder après que les détections les plus récurrentes.

Redimensionnements de l’image 
Le “découpage” de l’image en fragments


Exploitation des sous-détections
Des outputs du réseau de neurones aux sous-détections
A partir des outputs du réseau de neurones, nous souhaitons récupérer, pour chaque redimensionnement de l’image ainsi que sa taille originale, les scores “raffinés” (il est acceptable de le faire car il s’agit d’un problème de classification binaire). Un score raffiné est défini comme :
score pour la classe 1 - score pour la classe 0
Pour chaque score reçu, nous appliquons un softmax : les scores attribués aux classes 0 et 1 seront entre [0,1]. En conséquence le score raffiné sera entre [-1,1]. Si pour un fragment, le score raffiné est supérieur ou égal à un certain seuil (variable threshold_score dans configuration.py), nous définissons une “sous-détection” pour ce fragment.

Remarque: Les images ci-dessous sont données à titre illustratif et ne sont pas générées par l’algorithme.

Illustration des sous-détections pour différents redimensionnements de l’image. 
Dans notre exemple, on a choisi de redimensionner 2 fois (donc on a 3 images à slider). Il y a la représentation de 7 sous-détections avant la mise à l’échelle des sous-images.
Clustering & Filtrage
Notre réseau de neurones fait parfois de mauvaises classifications. Pour éliminer les “fausses” sous-détections, c’est-à-dire les sous-détections non redondantes, nous utilisons un système de clustering appliqués sur les centres des sous-détections. 
Dans notre cas, nous avons choisi DBSCAN qui regroupe les clusters en fonction de la densité.


Une identification des détections par DBSCAN

Cela permet de filtrer les sous-détections : on gardera uniquement les sous-détections formant un cluster c’est-à-dire qu’elle est représentée par un nombre suffisant de sous-détections (ce seuil est min_samples dans configuration.py). 

On choisit de garder les détections ayant au moins 2 sous-détections. 
En conséquence, on jette la détection 2, qui n’a qu’une sous-détection.


Remarque: Utiliser DBSCAN nous a permis de fusionner le clustering et le filtrage dans une seule fonction (get_detections).
Sélection de la meilleure sous-détection
Enfin, pour chaque détection, nous gardons la sous-détection avec le score le plus important.

L’affichage des résultats
Sur l’image originale, nous affichons les détections choisies (qui sont des carrés) en une couleur aléatoire ainsi que leur score.
Amélioration de performances du détecteur
Modification de epsilon
Une variable, n’a pas été citée comme hyperparamètre, et peut pourtant avoir beaucoup d’impact sur les résultats: eps de la méthode DBSCAN (de sklearn). Cela représente la distance de recherche de voisins pour un certain élément. Nous avons défini eps comme la moyenne des dimensions de l’image divisée par 100 (voir detections.py). 

Ex : Si l’image fait 1000x800, eps = (1000+800)/(2*100).
Impact des réductions successives de l’image par deux
Les réductions successives de l’image par deux après l’avoir réduit de div_factor n’est pas un problème.
En effet, sur les images que nous avons testées, nous avons vérifié manuellement que pour chaque visage, une sliding window à un redimensionnement donné était à une bonne taille.
Configurations : trouver les hyper-paramètres optimaux
Changer les configurations, situées dans configuration.py, peut modifier considérablement les résultats. Ce sont des hyper-paramètres autres que ceux du réseau de neurones.
Les variables influençant le nombre de détections
Les paramètres diminuant le nombre de sous-détections en augmentant : 
division_factor : plus l’image diminue rapidement de taille, plus on a de chance de sauter des sous-détections.
threshold_score : si le score de la sous-détection est trop faible, on considère que cette sous-détection n’est pas valable.
min_samples : le nombre de sous-détections nécessaire pour créer une détection.
stride : le pas pour la fenêtre glissante. Augmenter le pas va permettre de diminuer la consommation du temps et de la mémoire. Cependant, un pas trop important va entraîner le risque de “sauter”/”oublier” des visages (en particulier, pour les petites images).

Les paramètres augmentant le nombre de sous-détections en augmentant : 
nb_shrinkages : le nombre de fois ou une réduction de taille est appliquée sur l’image originelle . Cependant, cela augmente aussi la consommation du temps et de la mémoire.
Résultats obtenus

Après optimisation, voici les résultats que nous avons obtenu : 
Pour le training, le réseau de neurones assimile des images à raison de 300 images par seconde, ce qui est assez rapide mais tout de même problématique quand on doit traiter 100 000 images.
Pour le test avec l’ensemble de données fourni par le professeur, le réseau fait des prédictions pour 800 images par seconde. Les meilleurs résultats que nous ayons obtenus sont les suivants :
94% de réussite dans la détection des non-visages.
77% de réussite dans la détection des visages.
Pour un total de 93% de réussite dans les deux cas.
Pour l’application du réseau sur des images découpées par la sliding window, les résultats sont moins bons. Nous n’avons pas réussi à obtenir des résultats très satisfaisants de détection des visages, malgré le tuning des différents paramètres. Cependant, il ne détecte aucun visage sur une image représentant un paysage.

Résultats finaux pour différentes images :

Une image avec des détections, une image sans
Amélioration Algorithmique
	Une tentative d’utilisation sur le GPU a été faite sur un ordinateur, avec des résultats mitigés : l’utilisation du GPU était de 200Mo pour 16Go d’espace allouable sur le GPU. Cette performance n’est pas très concluante et n’a pas amélioré les performances dans notre cas.
Conclusion
	Nous pouvons faire différentes remarques : 
Il n’est pas facile de choisir les hyperparamètres, aussi bien pour le réseau de neurones (ex : learning rate, poids accordés à chaque classe) que pour la détection de visages dans une grande image (ex : le pas de décalage de la fenêtre glissante).
Pour l’étape de training, le fait d’ajouter une étape de data augmentation améliore considérablement les classifications des images mais est très coûteux en terme de temps.
Pour l’étape de testing sur une image quelconque, le fait de dupliquer l’image en plusieurs dimensions et de passer une sliding window sur chaque redimensionnement est très coûteux en terme de temps et mémoire. De plus, les détections ne sont pas très concluantes. Il est possible qu’il y ait des erreurs à une ou plusieurs fonctions du workflow, comme les fonctions pour calculer le centre des fragments d’images. Cependant, il est difficile d’avoir le temps pour des tests robustes sur les fonctions, et seuls de petits test ont été faits. 
Voies d’amélioration
Certaines configurations d’images ne seront certainement pas détectées, comme les images de profil, les visages (partiellement) cachés comme avec les lunettes de soleil…
Une meilleure utilisation du GPU serait extrêmement bénéfique en terme de temps.
Manuel de l’utilisateur

	Le fichier configuration.py permet de selectionner les hyperparamètres et d’autres paramètres de configuration (ex : le chemin jusqu’au jeu de données fourni par le projet de conception comme 'start_deep/start_deep/start_deep/', le fait d’appliquer ou non une data augmentation lors du training).

En lançant main.py, on lance notre programme. Après l'entraînement du réseau de neurones, notre programme propose deux possibilités de tests sous la forme de cette question : "Do you want to use the original testing dataset ? (yes/no)"
(=original testing dataset) → on peut tester notre réseau de neurones sur le jeu de tests venant du projet de conception. Cette option nous permet de calculer les performances du réseau de neurones. 
(=not original testing dataset) → on choisit de tester sur une grande image quelconque sur laquelle on souhaite détecter les visages. Le programme demande de spécifier le chemin vers le dossier contenant l’image. Le résultat est une sauvegarde de l’image d’origine avec les détections estimées (dans le dossier spécifié précédemment par l’utilisateur). (ex de chemin relatif à une image : own_dataset/ex3.jpg)
