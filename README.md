# project-template

# Goals for this repository
This repository exists for a few reasons. 
In general, it is part of our lab's ongoing effort to publish clean, consistent, and reproducible code in the pursuit of [Open Science](https://www.cos.io/).
More specifically, this repository serves three purposes:

1. It provides a starting point for lab members who may be new to organizing a Python data science project.
2. It provides examples of common data science tools and conventions that may be useful to lab members who are already comfortable with Python data science.
3. It provides a Bash script for generating a standard repository skeleton to help start new Python projects quickly.

The repository structure and workflow provided is **not intended to be presecriptive.**
There are obvious benefits to having a common repository structure, but [a foolish consistency is the hobgoblin of little minds](https://www.goodreads.com/quotes/353571-a-foolish-consistency-is-the-hobgoblin-of-little-minds-adored).
Every person has their preferred workflow, and every project has different needs.
Feel free to ignore/update/rename any component of this template to suit yours, and if you find a great tool/alternative workflow, let us know!

# Directory structure 
The directory structure is adapted from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).
```
.    
├── data    
│   ├── 01_raw   	    <--- all raw data goes here (and is NEVER edited directly) 
│   ├── 02_interim          <--- partly-processed/temporary data
│   └── 03_processed        <--- fully cleaned/processed data
├── models    		    <--- trained model fits
├── notebooks     	    <--- all scratch notebooks
│   └── jupytext            <--- used by jupytext, can be ignored by humans
├── reports     	    <--- markdown manuscripts/abstracts/etc 
│   └── figures     	    <--- finished figures (image files) used by reports
├── my_project    	    <--- the meat and potatoes (all code modules)
│   ├── __init__.py         <--- tells python that this is a module
│   ├── data    	    <--- code related to data processing/saving/loading
│   ├── models              <--- model code
│   ├── utils    	    <--- general-purpose code used around this repo
│   └── visualization  	    <--- plotting code
├── .gitignore  	    <--- tells git what not to track
├── README.md     	    <--- this file!
├── makefile    	    <--- useful general-purpose commands
├── requirements.txt        <--- lists the python packages used by this repo
└── setup.py     	    <--- used to install modules from my_project/
```

## Data

## Notebooks and Jupytext
Jupyter notebooks are incredibly useful for both exploratory analysis and results sharing.
There are many, many tutorials online ([this](https://www.youtube.com/watch?v=HW29067qVWk) is a good one), so this section will be restricted to some general workflow tips.

### Basic workflow tips
The template includes a directory called `notebooks/`.
[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) recommends treating this directory as your virtual lab notebook: numbering all entries ('01_myname_briefdescription.ipynb') and never deleting them.
Personally, I (Ben) like this workflow:

- All code (data munging, analyses, modelling, plotting, etc) begins in a jupyter notebook, which makes fast debugging easy.
Usually I try to stick to one task per notebook.
- Once my code is working the way that I like (and I know that I will reuse it), I'll stick it into a function or class, and copy it into the appropriate module folder.
For example, if I make a plotting function that I know I'll use many times, I'll put it into `my_project/visualization/vis.py`. 
Next time I need it, I can just import it (check out the [modules](#source-(my_project)) section for more info).
- My notebooks can end up being messy, and sometimes I'll break them as I update my codebase (for example, a notebook could use a plotting function that I have since changed).
Despite this, I never go back and update my notebooks- I like having a record of what I've done in case I need to revisit it.
If it's broken and I really need it to work, I can simply use [git](#using-git) to revert back to the state I was in when I first committed the notebook.
- Most of my notebooks begin with the same imports (`numpy`, `pandas`, `seaborn`, etc), as well as my own data-loading function.
Because I'm lazy, I put all of these imports/common cells into a notebook called `00_template.ipynb`.
When I'm creating a new notebook, I make a copy of the template with a quick terminal command: `$ cp notebooks/00_template.ipynb notebooks/0x_new.ipynb`.

### Version control
Jupyter notebooks are notoriously difficult to version-control.
This is because they're secretly just JSON files, so in addition to plaintext code they contain nasty stuff like JSON source code and big binary blobs:
```
"data": {
      "image/png": 
"iVBORw0KGgoAAAANSUhEUgAAAYwAAAEWCAYAAAB1xKBvAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAA
LEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8
vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJzsvXmcHNd13/s9vc4+2EgABHeQEkVSXGGRFLembFN
SPn7Wyy45i5UXh5ZjvcSy4xcr78WK5bwkzvKSeIllOqaVxZKcOJLN+FHc0dxJEVxAAgQBAiCIdbDP0tP
T+80fVdXdmOnl1q17ezBm/T6f+QDdXVXnVtU996z3HFFKESNGjBgxYvRDYrkHECNGjBgxVgZigREjRow
YMbQQC4wYMWLEiKGFWGDEiBEjRgwtxAIjRowYMWJoIRYYMWLEiBFDC7HAiBEDEJG/JiKPL/c4YsQ4nxE
LjBgfGojIXSLyoojMiMgZEXlBRH4IQCn1B0qp+x3QfExE/q+2z5tERHX5boNt+jFi2EQsMGJ8KCAiE8C
fAr8BrAE2Ab8ClB2Tfha4t+3zPcC7Hb57Tyk15XgsMWJEQiwwYnxY8BEApdS3lVJ1pdSCUupxpdRbACL
yRRF5PjjY1/i/JCLvichZEfktEZG23/8PEdnl//aYiFzWhe6zwJ0iEvDa3cC/A7Ys+u5Z/7qrReRPReS
kf+0/FZGL/d8+LyLb2i8uIl8RkYf9/2dF5F+LyEEROS4i3xCR4YjPLUaMJmKBEePDgj1AXUT+k4h8VkR
Wa5zzY8APATcCfwX4NICI/O/APwL+AnAB8Bzw7S7X+AGQ9a8BnjXxBLB30XfP+v9PAL8PXAZcCiwAv+n
/9jDwURG5uu36PwF8y///r+EJxpuAq/CsqF/WuM8YMbQQC4wYHwoopWaBuwAF/C5wUkQeFpH1PU77F0q
paaXUQWAr3kIM8NPAP1dK7VJK1YB/BtzUycpQSpWBV4B7RGQNsEoptR9PyATfXQs84x9/Win1P5RSRaX
UHPD/4ruvlFJF4E+ALwD4guMa4GHf+vk7wFeUUmf8c/8Z8HmzJxYjxlLEAiPGhwb+Av9FpdTFwPXARXj
uoW5ojykUgTH//5cB/15EpkVkGjgDCJ5G3wnP4lkRdwOB2+v5tu8OKaU+ABCRERH5HRH5QERm/XNXiUj
SP+9b+AIDz7r4Y1+QXACMAK+1jetR//sYMawgFhgxPpRQSr0LfBNPcITFIeCnlVKr2v6GlVIvdjn+WTz
BcA+eZQHwAnAn57qjAH4B+Chwm1Jqwv8dPIEE8DiwTkRuwhMcgTvqFJ776rq2MU0qpcaIEcMSYoER40M
BEblGRH6hLYB8Cd6C+7LB
 },
     "execution_count": 22,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ]
```
This is all hidden from you when you open the notebook in a browser or IDE, but when using [git](#using-git) this can be problematic.
Every tiny change to your notebook (even just re-running a cell without changing anything!) produces a massive change to the binary content, which git then detects as a massive change to the notebook.
These binary diffs are meaningless to a human reading the git commit, so most people prefer not to let git track their notebooks.

There are many workarounds for this (see [here](https://nextjournal.com/schmudde/how-to-version-control-jupyter)) for a thorough overview), but so far my favourite is a package called `jupytext`.
Jupytext works by creating a "pure" python version of your notebook without all the JSON/binary junk. 
Instead, Jupytext notebooks look like normal `.py` scripts, with a bit of metadata in a comment block at the top.
This means that 1) the output of notebook cells is not saved and 2) you can version control the Jupytext version of your notebooks and all git will see are changes to your actual code.

There are many ways to set Jupytext up (see their [documentation](https://jupytext.readthedocs.io/en/latest/introduction.html) if you want to get fancy), but the simplest and most IDE/editor agnostic way that I've found is implemented in the project template:

- The `notebooks/` directory contains a sub-directory `notebooks/jupytext/` that will contain the jupytext versions of all notebooks.
You never have to touch this directory (or the jupytext scripts inside).
- All files ending with `*.ipynb` are ignored by git (see [.gitignore](#gitignore)) so that no notebook files are tracked.
All `*.py` files in `notebooks/jupytext/` are tracked by git.
- To "link" all jupyter notebooks in `notebooks` to their Jupytext copies, use the command `$ jupytext --set-formats ipynb,jupytext//py --sync notebooks/*.ipynb`.
Now when you edit and save your local jupyter notebook via a jupyter notebook server in the browser, jupytext will automagically sync the changes to the linked python script.
- If you pull a repo with these scripts, you won't have the corresponding notebooks.
To update (or create) all notebooks from the scripts, use `$ jupytext --sync notebooks/jupytext/*.py`.

If this seems like a lot of work to manage your notebooks, you're right.
Personally, I keep those two jupytext commands in my `makefile` ([see below](#makefile)), and I've included them in the project template's Makefile as well.
When I want to make sure my notebook changes are tracked, I just run `$ make notebooks`, and everything is updated for me.

You can try this out for yourself to get a feel for it: when you first clone this repository to your machine, you won't have any notebooks in the `notebooks/` directory, but there are two Jupytext files in `notebooks/jupytext/`.
Run `$ make notebooks`, and the notebook equivalents should be generated!
Now, if you change the notebooks and save them, running `$ make notebooks` again will add those changes to the jupytext versions.

As a final test of this setup, try making a new notebook and running `$ make notebooks` again.


## Reports and Figures

## Source (my_project)

## Models

## Setup.py and requirements

## Using Git
When managing a project, it is important to use some form of version control. This allows you to track your changes, keep a backup of your work, and allows you to effectively collaborate with others. There are many version control solutions, but GitHub uses Git.

Watch the following tutorial for an introduction to git.   
[![](http://img.youtube.com/vi/HVsySz-h9r4/0.jpg)](http://www.youtube.com/watch?v=HVsySz-h9r4 "")

### Cloning a repository
When you want to download your own copy of the project, you have to clone the repository. There are several options when pressing the **"&downarrow; Code"** button:

* You can download a zip archive
* You can clone the repository with a Git command via `https`
* You can clone the repository with a Git command via `ssh`

Downloading a zip file is the simplest since you only require a web browser. However, to make the best use of Git you should have it installed on your workstation. Once you have git installed, using a terminal `cd` into your preferred directory and type the following command to clone this project:
```
$ git clone https://github.com/BlohmLab/project-template.git
```
You will be prompted to enter your username and password for private repositories.
```
Username for 'https://github.com':
Password for 'https://github.com':
```
When using https you will need to enter your username and password each time you want to make a change or update your copy of the project.

The benefit of using `ssh` is that you can setup a key that will allow your computer to be authenticated automatically.

#### Setting up SSH 

### The Git workflow

#### Branches
#### Commits
#### Pull requests

### .Gitignore
Git will track changes to all files added to the repository folder. This includes temporary files, or those created by software as a backup. You want to avoid adding unnecessary or irrelevant files to your repository. Fortunately Git provides a way to ignore certain types of files. A `.gitignore` file contains filenames and extensions that should not be tracked. For example, MacOS has an invisible `.DS_STORE` file in each directory that Git would include by default which is not a desired behaviour. You can prevent Git from tracking it by adding `.DS_STORE` on its own line in the `.gitignore` file. Any line that begins with a `#` will be interpreted as a comment. When using Jupyter Notebooks checkpoints are stored in a `.ipynb_checkpoints/` directory. We can tell git to ignore this directory by adding it to the `.gitignore` file.

## Makefile 
The top-level directory of this project contains a file called `makefile`.
This is not a python or shell script, but a unique text file that contains instructions (or "recipes") to be carried out by a GNU utility called `make`.
Makefiles are commonly used when writing C programs to help with code compilation.
This utility is installed by default on Linux systems, but can be installed with one command on macOS via [homebrew](https://formulae.brew.sh/formula/make) (`brew install make`) or Windows via [chocolatey](https://chocolatey.org/packages/make) (`choco install make`).

Python is an [interpreted](https://en.wikipedia.org/wiki/Interpreted_language) language that doesn't need to be compiled before running, so why would we use a C compilation utility for a Python data science project?
[This blog post](https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects) provides some motivation, but the short answer is "because it can make your life easier."

The quickest way to understand how a Makefile works is to `cd` into the top directory of the repository (where `makefile` lives) and type:

```
$ make
>> This is the first recipe in the file.
```

This should print the above text to your terminal. Let's try:

```
$ make hello
>> Hello!
```

Open `makefile` with a text editor and take a look. 
Ignore the first line (`.PHONY: ...`) for now, and instead check out the four "recipes" included in the makefile.
Recipes look (and operate) in an analagous way to functions in other languages.
We can call one of the recipes from a terminal by typing `make <recipe name>`.
In our case, the recipes simply contain Bash commands.
When we type `make hello`, the shell will execute `@echo "Hello!"` and we will see the output as if we had typed `echo "Hello!"` into our terminal.
The "@" prevents make from also printing the echo command back to us.

Let's try another recipe:

```
$ make all
>> This is the first recipe in the file.
```

This is the same output that we got when we typed `make`; make executes the first recipe in the file by default if you don't provide a recipe name.
In practice, it can be useful to have your first recipe simply list the various recipe options, so that you (or someone else) can quickly find a command without opening the makefile.

Recipes can be used for a lot more than just printing text.
In general, you can think of a Makefile as a place to store terminal commands or series of terminal commands that you want to execute fairly often, but can't be bothered to type out manually each time.
For example, this recipe can be used to sync Jupyter notebooks with Jupytext (see the [Jupytext](#notebooks-and-jupytext) section for more info):

```
notebooks: 
	@jupytext --set-formats ipynb,jupytext//py --sync notebooks/*.ipynb
	@jupytext --sync notebooks/jupytext/*.py
```

Instead of remembering and typing out these long commands every time you add a new notebook, you can plunk them into your Makefile and run them with `make notebooks`.

You can also use your Makefile to execute Python scripts.
For example, if you have a series of data processing scripts that must always be called in the same order, you could have a recipe like this:

```
data:
	@python data_script_1.py
	@echo "Step 1 complete."
	@python data_script_2.py
	@echo "Step 2 complete."
	@python data_script_3.py
	@echo "Step 3 complete."
```

The above examples should be enough to get you started using make to automate things.
A few closing notes:

- The `.PHONY: ...` line at the top of the Makefile simply tells make that the recipe doesn't reference a specific file (if we were using make for compilation, recipes would typically be asscociated with specific code). When you add a new recipe, add its name to this list so that make knows these are just general-purpose commands.
- Make interprets spaces and tabs differently! When indenting inside a recipe, make sure you use tab.
