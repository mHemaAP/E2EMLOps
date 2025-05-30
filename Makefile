help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo|.ruff_cache|.venv|\.egg-info|.pytest_cache|.ipynb_checkpoints|.ruff_cache|.gradio)" | xargs rm -rf
	rm -f .coverage
	find . -type f -name "uv.lock" -ls -delete
	

clean-logs: ## Clean logs
	find . -type f -name "train_log.log" -ls -delete
	find . -type f -name "eval_log.log" -ls -delete
	find multirun/** -type d -exec rm -rf {} + 2>/dev/null
	find logs/** -type d -exec rm -rf {} + 2>/dev/null
	find outputs/** -type d -exec rm -rf {} + 2>/dev/null
	find lightning_logs/** type d -exec rm -rf {} + 2>/dev/null


format: ## Run ruff hooks
	ruff check . --fix

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main


del-model:  ## Delete saved model
	find . -type f -name "*.onnx" -ls -delete
	find . -type f -name "*.pt" -ls -delete
	find . -type f -name "*.pth" -ls -delete
	find . -type f -name "*.ckpt" -ls -delete
	


#############################################################################################
# 
#		Tensorboard Dashboard
# 
#############################################################################################

struct: ## show tree structure
	tree -L 4 . --dirsfirst 

show: ## tensorboard logs on fastrun
	tensorboard --logdir logs/ --load_fast=false --bind_all --port 6006 &


showoff: ## turnoff tensorboard
	# kill -9 $(lsof -ti :6006)
	@PID=$$(lsof -ti :6006); \
	if [ -n "$$PID" ]; then \
		echo "Killing process $$PID"; \
		/usr/bin/kill -9 $$PID; \
	else \
		echo "No process found on port 6006"; \
	fi



#############################################################################################
# 
# 		SPORTS MODEL
# 
#############################################################################################
hsports:  ## hparam search on sports ds
	HYDRA_FULL_ERROR=1 python src/backend/torch_local/train.py experiment=hsports  -m
	echo "Best Hparams"
	cat multirun/*/*/optimization_results.yaml

tsports:  ## train model on sports ds
	HYDRA_FULL_ERROR=1 python src/backend/torch_local/train.py script=true name=sports callbacks.model_checkpoint.filename=sports

esports:  ## evaluate model on sports ds
	HYDRA_FULL_ERROR=1 python src/backend/torch_local/eval.py experiment=esports

gsports: ## gradio deploy on sports ds
	python src/frontend/gradios/space_sports.py 

fsports:  ## fastapi deploy on sports ds
	python src/backend/fastapi_app/fapi_sports.py 

tssports:
	echo "TORCHSERVE:: sports ds"


#############################################################################################
# 
# 		VEG & FRUITS MODEL
# 
#############################################################################################
hvegfruits:  ## hparam search on vegfruits ds
	echo "TODO: hparam search vegfruits"

tvegfruits:  ## train search on vegfruits ds
	python src/backend/torch_local/train.py model=mvegfruits data=dvegfruits script=true name=vegfruits callbacks.model_checkpoint.filename=vegfruits

evegfruits:  ## evaluate model on vegfruits ds
	HYDRA_FULL_ERROR=1 python src/backend/torch_local/eval.py experiment=evegfruits

gvegfruits:  ## gradio deploy on vegfruits ds
	python src/frontend/gradios/space_vegfruits.py 

fvegfruits:  ## fastapi deploy on sports ds
	python src/backend/fastapi_app/fapi_vegfruits.py 



#############################################################################################
# 
# 		GRADIO Deploy
# 
#############################################################################################
gradio-deploy:
	python K8SDeploy/AWSLambdas/app.py 