#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import ml

setup(

    # le nom de votre bibliothèque, tel qu'il apparaitre sur pypi
    name='ml',
    # la version du code
    version=ml.__version__,
    # Liste les packages à insérer dans la distribution
    # plutôt que de le faire à la main, on utilise la foncton
    # find_packages() de setuptools qui va cherche tous les packages
    # python recursivement dans le dossier courant.
    # C'est pour cette raison que l'on a tout mis dans un seul dossier:
    # on peut ainsi utiliser cette fonction facilement
    packages=find_packages(),

    # Une description longue, sera affichée pour présenter la lib
    # Généralement on dump le README ici
    long_description="""ml""",

    install_requires=["numpy", "scipy", "pandas>=0.16", "scikit-learn", "matplotlib"], #, "matplotlib"

    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

)