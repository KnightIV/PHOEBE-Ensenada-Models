import os
from collections import namedtuple

import phoebe

import general.utils as gen_utils

AdoptSolutionResult = namedtuple("AdoptSolutionResult", "solutionName computeModelName")
def adopt_solution(b: phoebe.Bundle, label:str=None, 
					reset_params=False, solution_file:str=None, adopt_twigs:list[str]=None,
					run_compute=True, print_sol=True, compute='phoebe01', **compute_kwargs) -> AdoptSolutionResult:
	solutionName: str
	if label is not None:
		solutionName = f"opt_{label}_solution"

	if solution_file:
		solutionName = b.import_solution(solution_file, overwrite=True).solutions[0]
		label = solutionName.replace("_solution", "").replace("opt_", "")

	if print_sol:
		print("Adopted:")
		gen_utils.printFittedVals(b, solutionName, adopt_twigs=adopt_twigs)
		print("\nOriginal values:")
		gen_utils.printFittedTwigsConstraints(b, solutionName, adopt_twigs=adopt_twigs)

	try:
		initValues = {}
		if reset_params:
			for twig in b.get_value(qualifier='fitted_twigs', solution=solutionName):
				initValues[twig] = b.get_quantity(twig)

		b.adopt_solution(solutionName, adopt_parameters=adopt_twigs)

		computeModelName = None
		if run_compute: 
			computeModelName = f"opt_{label}_model"
			b.run_compute(model=computeModelName, compute=compute, **compute_kwargs, overwrite=True)
	except: # reset values if an exception occurs, regardless of reset_params value
		for twig, val in initValues.items():
			b.set_value(twig, value=val)
	finally:
		if reset_params:
			for twig, val in initValues.items():
				b.set_value(twig, value=val)
	
	return AdoptSolutionResult(solutionName, computeModelName)

def optimize_params(b: phoebe.Bundle, fit_twigs: list[str], label: str, export: bool, datasets: list[str], subfolder: str=None, 
					optimizer='optimizer.nelder_mead', compute='phoebe01', overwrite_export=True,
					**solver_kwargs):
	if not 'maxiter' in solver_kwargs.keys():
		solver_kwargs['maxiter'] = 200 if export else 10

	abilitatedDatasets = [d for d in b.datasets if b.get_value(qualifier='enabled', dataset=d)]
	gen_utils.abilitateDatasets(b, datasets, False)
	
	saveIterProgress = 1 if export and optimizer == 'optimizer.nelder_mead' else 0
	b.add_solver(optimizer, solver=f'opt_{label}', fit_parameters=fit_twigs, overwrite=True, progress_every_niters=saveIterProgress, compute=compute, **solver_kwargs)
	if export:
		if not os.path.exists('external-jobs'):
			os.mkdir('external-jobs')
		if subfolder is not None:
			os.makedirs(os.path.join('external-jobs', subfolder), exist_ok=True)
		
		exportPath = f'./external-jobs{f"/{subfolder}" if subfolder is not None else ""}/{optimizer}_opt_{label}.py'
		if not overwrite_export and os.path.exists(exportPath):
			print("Solver already exists |", exportPath)
		else:
			fname, out_fname = b.export_solver(script_fname=exportPath, out_fname=f'./results/opt_{label}_solution', 
												solver=f'opt_{label}', solution=f'opt_{label}_solution', overwrite=True)
			print("External Solver:", fname, out_fname)
	else:
		b.run_solver(solver=f'opt_{label}', solution=f'opt_{label}_solution', overwrite=True, **solver_kwargs)

	gen_utils.abilitateDatasets(b, abilitatedDatasets)
	
	return f'opt_{label}', f'opt_{label}_solution'