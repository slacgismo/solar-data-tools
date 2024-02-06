# Prepare your data plug
Here is the basic usage of the example data plugs and instruction on how to write you own.

**Note that if the data plug depends on any packages that are not in the Python standard library, they should be included in a requirements.txt file that accompanies the data plug.**

[TODO] Fill out info about example data plugs and how users can write their own.

[TODO] Make a standard attribute for data plugs: a tuple that has one, two or more values corresponding to the inputs that each data plug requires: 
- CSV plug: list of file names
- PVDAQ plug: files IDs and associated years
- PVDB (Redshift) plug: list of site IDs and list of inverter index numbers
	- needs list of non-standard libraries in requirements file (```requests``` only so far)