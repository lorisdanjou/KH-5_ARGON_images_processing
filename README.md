# KH5-ARGON images ortho-correction and georeferencing

## TODO
### Ortho-correction
- [ ] change name of xc, yx in `geometry.internal_orientation`.
- [x] clean `geometry/internal_orientation.py`: choose one method (Molnar?) and remove the others.
- [x] in `geometry.internal_orientation.py`, rename and/or move `objective_function`
- [x] implement lens distortion
- [ ] implement inverse lens distortion
- [x] implement & test global objective function as in _Molnar et al._, for one image only. → not fully satisfactory
- [x] implement & test global objective function as in _Molnar et al._, for several images. → not fully satisfactory
- [ ] double checks all units (change all to m?) and coordinates transformation (especially external orientation).
- [ ] finish documenting the notebooks.
- [ ] implement RCP coefficients retrieval w/ least squares.
- [ ] ortho-correct the images w/ GDAL.
- [ ] when there is a repetition of the same commands in a notebook, pack everything in functions.
### Simple georeferencing
- [ ] split images and GCPs in smaller tiles to make georeferencing easier.