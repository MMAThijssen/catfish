from sys import argv
import h5py 
import os

def get_baselength(fast5, use_tombo=True):
    """
    Takes events to extend base sequence per measurement.
    
    Args:
        fast5 - str, path to FAST5
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
        
    Returns: list of str
    """
    # maybe should be in TrainingRead and rebuild val and testdb
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_bases = hdf[hdf_events_path]["base"].astype(str)
        
    return len(event_bases)
    
if __name__ == "__main__":
    #~ sgl_folder = argv[1]
    #~ sgl_files = os.listdir(os.path.abspath(sgl_folder))
    #~ os.chdir(sgl_folder)
    #~ summed = 0
    #~ for sgl in sgl_files:
        #~ print(sgl)
        #~ s = get_baselength(sgl)
        #~ summed += s
    #~ print(summed)
    print(get_baselength(argv[1]))
