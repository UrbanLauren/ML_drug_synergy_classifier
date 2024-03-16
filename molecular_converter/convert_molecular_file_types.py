#import packages
import getopt
import matplotlib.pyplot as plt
import numpy as np
import openbabel
import os
import pandas as pd
import pprint
import pybel
import re
import rdkit
from rdkit import Chem, rdBase, RDConfig  #only works on python mypy36env
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, AllChem, PandasTools
from rdkit.Chem.rdmolfiles import SmilesWriter, SDWriter
import subprocess
import seaborn as sns
import sys
import traceback

# Define Variables
mol = openbabel.OBMol()

def convert_aa_seq_to_smiles(df, col_name_aa_seq):
    # create a list from column "sequence" in dataframe "data" that contains AA sequences
    aa_seq = (df[col_name_aa_seq].to_list())
    # Create the blank column to add outputs to
    df["smiles"]= ""
    #for loop to convert the AA sequence (listed in column ) into smiles. If error return "error" and add results as new column in dataset "data"
    for i in range(len(aa_seq)):
        try:
            results = Chem.MolToSmiles(Chem.MolFromFASTA(aa_seq[i]))
            df.at[i, "smiles"] = results
        except:
            df.at[i, "smiles"] = "error"
    return df


def convert_smiles_to_sdf(input_file, output_sdf_filename):
    try:
        PandasTools.AddMoleculeColumnToFrame(input_file, smilesCol='smiles', molCol='SDF')
        sdf_file = input_file[['ID', 'NAME', 'SEQUENCE', 'smiles', 'SDF']].drop_duplicates(subset=['SEQUENCE'])
        PandasTools.WriteSDF(sdf_file, output_sdf_filename, molColName='SDF', idName = 'ID', properties=None)
        print(f"file with SDF column saved here {output_sdf_filename}")
    except:
        print("Error in converting, check smiles and ID column names in input_file")
    return sdf_file
    
    
# Write outputs of molecular conversion to file (string type)
def write_contents(filename, contents):
  # do some basic checking, could use assert strictly speaking
  assert filename is not None, "filename cannot be None"
  assert contents is not None, "contents cannot be None"
  f = open(filename, "w")
  f.write(contents)
  f.close() # close the file
  

#Write a list to a file
def write_list_to_file(filename, list, line_sep = os.linesep):
  # do some basic checking, could use assert strictly speaking
  assert list is not None and len(list) > 0, "list cannot be None or empty"
  write_contents(filename, line_sep.join(list))
  

class InputError(Exception):
    pass

def smiles_to_sdf(smiles):
    try:
        mymol = pybel.readstring('smi', smiles)
    except:
        print("Unexpected error:", sys.exc_info())
        traceback.print_tb(sys.exc_info()[2])
        raise InputError
    try:
        mymol.draw(show=False, update=True)
    except:
        pass
    return mymol.write(format='sdf')


def sdf_to_smiles(sdf):
    # dos2unix
    sdf = sdf.split('\r\n')
    sdf = '\n'.join(sdf)
    sdf = sdf.split('\n')
    sdf[1] = ''
    sdf = '\n'.join(sdf)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats('sdf', 'smi')
    mol = openbabel.OBMol()
    if not obConversion.ReadString(mol, sdf):
        raise InputError
    return obConversion.WriteString(mol)


def batch_smiles_to_sdf(smiles):
    buf = ''
    err = 0
    for s in smiles.splitlines():
        try:
            buf += smiles_to_sdf(s)
        except InputError:
            err += 1
    return (buf, err)
