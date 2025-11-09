## Build Environment

```bash
sudo apt install python3-virtualenv
virtualenv venv
source ./venv/bin/activate

pip3 install cvxpy,numpy,matplotlib
pip3 install "cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,SCS,CLARABEL,QOCO,ECOS]"
```

## Run

```bash
python3 xxx.py
```