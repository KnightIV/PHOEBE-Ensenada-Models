import sys
import os
import gzip
import shutil

import numpy as np
import phoebe
from phoebe import u

try:
	from utils import printFittedVals, printChi2
except ImportError: # will happen when running on external compute, copy over necessary functions here
	def __matchAnyTwig(twig: str, twigs_list: list[str]) -> bool:
		for refTwig in twigs_list:
			refComponents = refTwig.split('@')
			twigComponents = twig.split('@')

			if len(set(refComponents) & set(twigComponents)) != 0:
				return True
			
		return False
	
	def printFittedVals(b: phoebe.Bundle, solution: str, adopt_twigs: list[str] = None, units: dict[str, u.Unit] = {}):
		for twig, value, unit in zip(b.get_value('fitted_twigs', solution=solution),
									b.get_value('fitted_values', solution=solution),
									b.get_value('fitted_units', solution=solution)):
			try:
				originalUnit = u.Unit(unit)
				quantity = value * originalUnit
				print(twig, 
						f"{quantity.to(units.get(twig, originalUnit)).value:.5f}", 
						units.get(twig, originalUnit).to_string(), 
						"(Not adopting)" if adopt_twigs is not None and not __matchAnyTwig(twig, adopt_twigs) else "")
			except:
				print(twig, value, unit)
	
	def printChi2(b: phoebe.Bundle, model: str):
		"""
		Prints the chi2 fit of a model for all available datasets: Iturbide, Aviles, Gaia, and ZTF data, both normalized
		and raw datasets. Silently ignores any dataset that isn't present in the specified model.
		"""

		ztfDatasets = [d for d in b.datasets if 'Ztf' in d]
		spmDatasets = [d for d in b.datasets if 'Spm' in d]
		
		print(model, "=================================================", sep='\n')

		try:
			print('\t', "TESS -", np.sum(b.calculate_chi2(model=model, dataset='lcTess')))
		except: pass

		print("------------------------------------------------")

		try:
			print('\t', "OAN SPM -", np.sum(b.calculate_chi2(model=model, dataset=spmDatasets)))
			for d in spmDatasets:
				print('\t\t', d, "-", np.sum(b.calculate_chi2(model=model, dataset=d)))
		except: pass

		print("------------------------------------------------")

		try:
			print('\t', "ZTF -", np.sum(b.calculate_chi2(model=model, dataset=ztfDatasets)))
			for zd in ztfDatasets:
				try:
					print('\t\t', zd, "-", np.sum(b.calculate_chi2(model=model, dataset=zd)))
				except: 
					print("\t\t", zd, "Not found in model")
		except:
			pass

def load_bundle(path: str) -> phoebe.Bundle:
	tempJsonFile = path.replace('.gz', '') # work with compressed files
	with gzip.open(path, 'rb') as f_in:
		with open(tempJsonFile, 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
	
	b = phoebe.load(tempJsonFile)
	os.remove(tempJsonFile)
	return b

def save_bundle(b: phoebe.Bundle, path: str, compact: bool = True, compress: bool = True) -> str:
	if '.json' not in path:
		path = f"{path}.json"
		
	jsonFile = b.save(path, compact=compact)
	if compress:
		with open(jsonFile, 'rb') as f_in:
			with gzip.open(f"{jsonFile}.gz", 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		os.remove(jsonFile)
		return f"{jsonFile}.gz"
	
	return jsonFile

def run_dc(b: phoebe.Bundle, num_iter: int, solver: str, solution: str) -> None:
	"""
	Run differential corrections algorithm for the specified number of iterations.

	Final optimizer solution saved as whole bundle.
	"""
	for i in range(num_iter):
		print('', i, "-------------------------", sep='\n')
		b.run_solver(solver=solver, solution=solution, overwrite=True)
		printFittedVals(b, solution=solution)
		b.adopt_solution(solution)

		b.run_compute(model='dc_solution_model', overwrite=True)
		printChi2(b, model='dc_solution_model')

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print("Usage: python dc-optimizer.py {solver} {solution} {num_iter} {bundle_start_path} {result_path}")
		exit(1)

	logger = phoebe.logger(clevel='WARNING')

	# arguments expected to script
        # solver name
		# solution name
		# bundle path to run corrections on
		# number of iterations to perform differential corrections
	_, solver_name, solution_name, num_iter, bundle_start, result_path = sys.argv

	# b: phoebe.Bundle = phoebe.load(bundle_start)
	b = load_bundle(bundle_start)
	run_dc(b, num_iter=int(num_iter), solver=solver_name, solution=solution_name)
	# b.save(result_path, compact=True)
	save_bundle(b, result_path)
	