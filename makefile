.PHONY: all hello notebooks data 

all: 
	@echo "This is the first recipe in the makefile."

hello:
	@echo "Hello!"

notebooks: 
	@jupytext --set-formats ipynb,jupytext//py --sync notebooks/*.ipynb
	@jupytext --sync notebooks/jupytext/*.py

data:
	@python data_script_1.py
	@echo "Step 1 complete."
	@python data_script_2.py
	@echo "Step 2 complete."
	@python data_script_3.py
	@echo "Step 3 complete."
	
newnotebook:
	@cp notebooks/00_template.ipynb notebooks/0x_new.ipynb
