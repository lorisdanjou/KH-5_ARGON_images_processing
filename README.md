# KH5-ARGON images ortho-correction and georeferencing

## TODO
- [ ] clean `geometry/internal_orientation.py`: choose one method (Molnar?) and remove the others.
- [ ] in `geometry.internal_orientation.py`, rename and/or move `objective_function`
- [x] implement lens distortion
- [ ] implement lens distortion inversion
- [x] implement & test global objective function as in _Molnar et al._, for one image only. → not satisfactory
- [x] implement & test global objective function as in _Molnar et al._, for several images. → not satisfactory
- [ ] split images and GCPs in smaller tiles to make georeferencing easier.