# Solar Data Tools Dask Tool

## Dataplugs
All dataplugs must have an associated `requirements.txt` file if they 
depend on any packages that are not included in the Python Standard Library. 
This file should be available to install on the container being created for  
running on the cloud compute clients. For example, these lines would be 
added to the Dockerfile associated with the Docker image to be used on AWS 
Fargate or Azure VMs:
  ```dockerfile
  WORKDIR /root
  RUN mkdir sdt
  WORKDIR /root/sdt

  COPY dataplug_requirements.txt /root/sdt/.
  COPY requirements.txt /root/sdt/.
  
  RUN pip install -r dataplug_requirements.txt
  RUN pip install -r sdt_requirements.txt
  ```
Note that the SDT package's `requirements.txt` file needs to be installed 
_after_ the plug's requirements to avoid running into version issues with SDT. 
The 
users should be made aware through proper documentation of SDT's 
dependencies and any package versions that may cause issues if up/downgraded.

### Finding edge cases 
We want to find edge cases where this method might fail (for example trying 
to install older versions of `numpy` or other common packages with a 
`dataplug_requirement.txt`).
We want to:
- Find cases where this method of creating containers will fail
- Document what errors look like from the user's perspective
- Implement error handling of any exceptions (errors should make it clear to users that there's a dependency version issue)

### Examples for users
Each existing dataplug should have at least a `dataplug.py` module, along 
with a `requirements.txt` file if needed. The documentation should clearly 
show users how to create their own plugs (they must provide these two files) 
using these examples.


## Clients
 
Each client should have a `client.py` example module containing a class that 
will include all user info (authentication, compute info, etc). The SDT Dask 
tool should interact with all clients using the same method to maintain 
seamless interoperability (similar to how `dataplug.get_data` works).
The  users need to provide this `.py` file and should be able to produce 
their own based on our examples and documentation.


## SDT Dask Tool
The tool needs to be designed with the user in mind: think of how people are 
going to interact with it (CLI, python kernel, notebooks, scripts, etc). 
This will define certain methods, for example: how do you load the 
user's files. 

The tool should get all the needed information from the user files
(`client.py` and `dataplug.py`) including any required environment variables 
(which we should clearly define in the documentation).

Example tool definition:
```python
class SDTDask:
    def __init__(self, path_to_user_files):
        # Finds client.py, dataplug.py, requirements.txt (if any)
        # provided by user and instantiates Client and DataPlug
        # based on the standard methods that we set
```

Example usage by user:
```python
from solardatatools import SDTDask

# Instantiate
dask_tool = SDTDask(path_to_my_files)

# Keys to run pipeline on 
# Alternatively, these can be provided in dataplug.py
keys = [(12, 2011), (13, [2011,2012]) 

# Run pipeline on provided client
dask_tool.execute(keys)
```

Handling of exceptions should also be thought through (for example errors 
from dependencies while running, see above), including how to handle pipeline 
crashes in the tool (e.g. how will the user know what happened?)


## Open Questions
- Does the user create the container ahead of time (using both requirements 
  files) and then set up the container name in the `client.py`? In this case,
  we may not need to have the SDT Dask tool have access to the 
  `dataplug_requirements.txt` file.
