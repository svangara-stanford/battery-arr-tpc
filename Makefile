.PHONY: demo inspect-author-model attia-reference-smoke attia-reference-exact agent-rediscovery-offline hackathon-demo

demo:
	python -c "import battery_aar; print('battery_aar demo import ok')"

inspect-author-model:
	python scripts/inspect_oed_mat_model.py --model literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model.mat

attia-reference-smoke:
	python scripts/run_attia_reference_reproduction.py --smoke --allow-partial --skip-validation-batch

attia-reference-exact:
	python scripts/run_attia_reference_reproduction.py \
		--battery-fast-charging-root literature_models_and_data/battery-fast-charging \
		--out runs/attia_reference_reproduction \
		--reports-dir reports \
		--require-exact-author-model \
		--skip-validation-batch

agent-rediscovery-offline:
	python scripts/run_agentic_rediscovery.py --offline --agents 2 --iterations 2 --out runs/open_battery_agents/offline_smoke --reports-dir reports

hackathon-demo:
	pytest
	$(MAKE) demo
	$(MAKE) attia-reference-smoke
	$(MAKE) agent-rediscovery-offline
