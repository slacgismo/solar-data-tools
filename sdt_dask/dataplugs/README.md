# Prepare your data plug
Here is the basic usage of the example data plugs and instruction on how to write you own.

**Note that if the data plug depends on any packages that are not in the Python standard library, they should be included in a requirements.txt file that accompanies the data plug.**

[TODO] Fill out info about example data plugs and how users can write their own.

[TODO] Make a standard attribute for data plugs: a tuple that has one, two or more values corresponding to the inputs that each data plug requires: 
- CSV plug: file name
- PVDAQ plug: files ID and associated year(s)
- PVDB (Redshift) plug: site ID and list of inverter index numbers

[TODO]
- Need to include requirements for each dataplug, which should be a list of any non-standard libraries (e.g. in a requirements file)
