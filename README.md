# SO/PHI Data Tutorial

This repo originally contained the SO/PHI tutorial for the Data Analysis Day at the Solar Orbiter 8th Workshop in Belfast UK (September 2022).

The original form can be found here: https://github.com/SolarOrbiterWorkshop/solo8_tutorials

The tutorial has now been updated (October 2023)

<img src="./static/philogo-1.png" width="220" align="left"/>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Useful Links and Information

- [SO/PHI Data Releases and Publication](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi/data-releases)
- [SO/PHI-FDT Quick Look Data](https://www.uv.es/jublanro/phidata_fdt.html)
- [SO/PHI-HRT Quick Look Data](https://www.uv.es/jublanro/phidata_hrt.html)
- [SO/PHI Data Format and Tutorials](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi/data)
- [SO/PHI Instrument Information](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi)
- [SOAR](https://soar.esac.esa.int/soar/#search)
- [SOAR Inventory Plots](https://www.cosmos.esa.int/web/soar/inventory-plots)
- [SOAR Python Download Github Gist](https://gist.github.com/JonasSinjan/e10053b972e5fb72057c078c7c275a5e)


## Instrument Basics and Key properties

- Two telescopes:
  - HRT (High Resolution Telescope)
  - FDT (Full Disc Telescope)
- FDT always has full disc within FOV
- HRT has much smaller FOV with much higher resolution
- They cannot operate simultaneously
- They are not continuously operating -> check quick look links above to see when Data is available
- They work in a very similar way to SDO/HMI -> for a comparison see: [SDO/HMI - SO/PHI-HRT Comparison](https://doi.org/10.1051/0004-6361/202245830)
- If you have ideas for scientific campaigns - get in contact via: `sophi_support@mps.mpg.de`

![Alt Text](./static/sophi_fov_rotateSun_new.gif)


## Setup and Installation

Tested OS:
- Linux Ubuntu

```bash=
conda env create --name phi_tutorial --file=phi_tutorial.yml
```

Otherwise manually install packages listed in `requirements.txt` with pip

### Troubleshooting

If the environment `phi_tutorial` does not appear as a kernel in Jupyter Notebooks:

```bash=
conda activate phi_tutorial
pip install ipykernel
python -m ipykernel install --user --name phi_tutorial --display-name "Python (phi_tutorial)"
```
