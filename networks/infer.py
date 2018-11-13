#!/usr/bin/env python3

### FOR INFERING:
# 1. Open fast5
# 2. Extract raw signals
# 3. Process like done in TrainingRead:
#       - cut part till first sample
#       - normalize signal
# 4. Save as npz file
# 5. Predict


    @raw.setter
    def raw(self, _):
        if self.use_tombo:
            first_sample = self.hdf['{hdf_path}BaseCalled_template/Events'.format(hdf_path=self.hdf_path)].attrs[
                'read_start_rel_to_raw']
        else:
            first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        raw_varname = self.hdf['Raw/Reads/'].visit(str)
        raw = self.hdf['Raw/Reads/'+ raw_varname + '/Signal'][()]
        raw = raw[first_sample:]
        self._raw = normalize_raw_signal(raw, self.normalization)

    np.savez(npz_path + splitext(basename(files))[0],
             base_labels=tr.classified, 
             raw=tr.raw[: tr.final_signal])
