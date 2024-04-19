import os

import phoebe
from phoebe import u

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matplotlib.animation import FuncAnimation
from IPython import display
import ipywidgets

SPM_COLORS = {
    'lcSpmJohnsonB@dataset': 'b',
	'Johnson:B': 'b',
    'lcSpmJohnsonB@model': 'darkblue',

    'lcSpmJohnsonV@dataset': 'g',
	'Johnson:V': 'g',
    'lcSpmJohnsonV@model': 'darkgreen',

    'lcSpmCousinsR@dataset': 'r',
    'Cousins:R': 'r',
    'lcSpmCousinsR@model': 'darkred',

    'lcSpmCousinsI@dataset': 'orange',
    'Cousins:I': 'orange',
    'lcSpmCousinsI@model': 'goldenrod'
}

TESS_COLORS = {
	'lcTess@dataset': 'lightblue',
	'lcTess@model': 'blue'
}

ZTF_COLORS = {
	'lcZtfR@dataset': 'r',
	'lcZtfR@model': 'darkred',

	'lcZtfG@dataset': 'g',
	'lcZtfG@model': 'darkgreen',

	'lcZtfI@dataset': 'orange',
	'lcZtfI@model': 'goldenrod',

	# trimmed datasets
	'lcZtfRTrimmed@dataset': 'r',
	'lcZtfRTrimmed@model': 'darkred',

	'lcZtfGTrimmed@dataset': 'g',
	'lcZtfGTrimmed@model': 'darkgreen',

	'lcZtfITrimmed@dataset': 'orange',
	'lcZtfITrimmed@model': 'goldenrod'
}

def displayAnims(rows: int, cols: int, *anims: FuncAnimation):
	plt.rcParams["animation.html"] = "html5"
	plt.rcParams["figure.figsize"] = (15,8)
	
	grid = ipywidgets.GridspecLayout(n_rows=rows, n_columns=cols)
	for row in range(rows):
		for col in range(cols):
			index = (row*cols) + col
			anim = anims[index]
			grid[row, col] = ipywidgets.HTML(anim.to_html5_video())

	display.display(grid)

def displayAnim(anim: FuncAnimation):
	originalBackend = plt.rcParams['backend']
	plt.rcParams['backend'] = 'Agg'
	display.display(display.HTML(anim.to_html5_video()))
	plt.rcParams['backend'] = originalBackend

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
			print(twig, f"{quantity.to(units.get(twig, originalUnit)).value:.5f}", units.get(twig, originalUnit).to_string(), "(Not adopting)" if adopt_twigs is not None and not __matchAnyTwig(twig, adopt_twigs) else "")
		except:
			print(twig, value, unit)

def printFittedTwigsConstraints(b: phoebe.Bundle, solution: str, units: dict[str, u.Unit] = {}, adopt_twigs: list[str] = None):
	for fitTwig in b.get_value('fitted_twigs', solution=solution):
		quantity = b.get_quantity(fitTwig)
		print("C" if b[fitTwig].constrained_by else " ", 
				fitTwig, 
				f"{quantity.to(units.get(fitTwig, quantity.unit)):.5f}", 
				"(Not adopting)" if adopt_twigs is not None and not __matchAnyTwig(fitTwig, adopt_twigs) else "")

def saveBundle(b: phoebe.Bundle, bundleName: str, subfolder: str = None, overwrite: bool = True) -> str:
	if not os.path.exists("bundle-saves"):
		os.mkdir("bundle-saves")

	saveFolder = "bundle-saves"
	if subfolder:
		saveFolder = f"bundle-saves/{subfolder}"
		os.makedirs(saveFolder, exist_ok=True)

	if os.path.exists(os.path.join(saveFolder, bundleName)):
		if overwrite:
			print(f"CAUTION: overwriting {os.path.join(saveFolder, bundleName)}")
		else:
			print(f"NOT OVERWRITING: {os.path.join(saveFolder, bundleName)} bundle already exists.")
	
	return b.save(f"{saveFolder}/{bundleName}")

def genAnimatedMesh(b: phoebe.Bundle, logger=None, meshDataset="mesh01", fc='teffs', **plot_kwargs):
	if logger: logger.setLevel('ERROR')
	default_kwargs = {
		"draw_sidebars": True,
		"color": "inferno"
	}

	_, mplfig = b.plot(dataset=meshDataset, kind='mesh', fc=fc, ec='face', animate=True, **(default_kwargs | plot_kwargs))
	if logger: logger.setLevel('WARNING')
	return mplfig

def animateMesh(b: phoebe.Bundle, logger=None, meshDataset="mesh01", fc='teffs', **plot_kwargs):
	displayAnim(genAnimatedMesh(b, logger, meshDataset, fc, **plot_kwargs))

def getEnabledDatasets(b: phoebe.Bundle):
	enabledDatasets = []
	for c in b.computes:
		for d in b.datasets:
			if b.get_value(qualifier='enabled', compute=c, dataset=d) and d not in enabledDatasets:
				enabledDatasets.append(d)
	return enabledDatasets

def abilitateDatasets(b: phoebe.Bundle, enableDatasets: list[str], includeMesh: bool = True):
	"""
	Enables specified datasets and disables all others.
	"""
	enableDatasets = enableDatasets.copy()
	if includeMesh:
		enableDatasets.append('mesh01')
		
	for d in b.datasets:
		if d in enableDatasets:
			b.enable_dataset(d)
		else:
			b.disable_dataset(d)
		# b.set_value_all(qualifier='enabled', dataset=d, value=(d in localDatasets))

def plotFigSize(b: phoebe.Bundle, figsize: tuple[float, float], **plot_kwargs):
	fig = plt.figure(figsize=figsize)
	b.plot(fig=fig, **plot_kwargs)

def plotModelResidualsFigsize(b: phoebe.Bundle, figsize: tuple[float, float], datasetGroups: list[list[str] | str], model: str, 
							  model_kwargs: dict['str', 'str'] = {}, residuals_kwargs: dict['str', 'str'] = {}, **plot_kwargs) -> dict[str, Figure]:
	"""
	Plots specified model for the datasets given. Plots dataset(s) with model overlay alongside residuals side-by-side.
	"""
	defaultPlotKwargs = {
		'marker': {'dataset': '.'},
		'color': SPM_COLORS | TESS_COLORS | ZTF_COLORS,
		'legend': True,
		'ls': {'model': 'solid'}
	}
	for key, defaultVal in defaultPlotKwargs.items():
		plot_kwargs[key] = plot_kwargs.get(key, defaultVal)

	residuals_kwargs['marker'] = '.'

	datasetGroupsFigures = {}
	for datasets in datasetGroups:
		maxFlux = 0
		for d in datasets:
			maxFlux = max([maxFlux, max(b.get_value(qualifier='fluxes', context='dataset', dataset=d))])
		maxFluxScale = 1 + 0.12*(len(datasets))

		fig = plt.figure(figsize=figsize)
		b.plot(x='phase', model=model, dataset=datasets, axorder=1, fig=fig, s={'dataset':0.008, 'model': 0.01}, ylim=(None, maxFluxScale*maxFlux), **(plot_kwargs | model_kwargs))
		b.plot(x='phase', y='residuals', model=model, dataset=datasets, axorder=2, fig=fig, subplot_grid=(1,2), s=0.008, show=True, **(plot_kwargs | residuals_kwargs))
		datasetGroupsFigures["-".join(datasets)] = fig
	return datasetGroupsFigures

def exportCompute(b: phoebe.Bundle, model: str, datasets: list[str], subfolder: str = None, **compute_kwargs) -> None:
	if not os.path.exists("external-compute"):
		os.mkdir("external-compute")

	computeFolder = "external-compute"
	if subfolder:
		computeFolder = f"external-compute/{subfolder}"
		os.makedirs(computeFolder, exist_ok=True)

	b.export_compute(script_fname=os.path.join(computeFolder, f"{model}.py"), out_fname=os.path.join(computeFolder, "results", f"{model}.model"), 
				  model=model, dataset=datasets, **compute_kwargs)
	
def adopt_solution(b: phoebe.Bundle, solution_name:str=None, model_name: str = None,
					reset_params=False, solution_file:str=None, 
					run_compute=True, print_sol=True, print_units={}, compute='phoebe01', **compute_kwargs) -> None:
	if solution_file:
		solution_name = b.import_solution(solution_file, solution=solution_name, overwrite=True).solutions[0]

	if print_sol:
		print("Adopted:")
		printFittedVals(b, solution_name, units=print_units)
		print("\nOriginal values:")
		printFittedTwigsConstraints(b, solution_name, units=print_units)

	try:
		initValues = {}
		if reset_params:
			for twig in b.get_value(qualifier='fitted_twigs', solution=solution_name):
				initValues[twig] = b.get_quantity(twig)

		b.adopt_solution(solution_name)

		if run_compute: 
			b.run_compute(model=model_name, compute=compute, **compute_kwargs, overwrite=True)
	except:
		if reset_params:
			for twig, val in initValues.items():
				b.set_value(twig, value=val)
	finally:
		if reset_params:
			for twig, val in initValues.items():
				b.set_value(twig, value=val)


def printChi2(b: phoebe.Bundle, model: str):
	"""
	Prints the chi2 fit of a model for all available datasets: Iturbide, Aviles, Gaia, and ZTF data, both normalized
	and raw datasets. Silently ignores any dataset that isn't present in the specified model.
	"""

	# rawGaiaDatasets = [d for d in b.datasets if 'raw' in d and 'gaia' in d]
	# normGaiaDatasets = [d for d in b.datasets if 'norm' in d and 'gaia' in d]
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

def printAllModelsChi2(b: phoebe.Bundle, filters: list[str] = []):
	for m in b.models:
		for f in filters:
			if f in m:
				printChi2(b, m)