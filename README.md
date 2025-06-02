# KH5-ARGON images ortho-correction and georeferencing

## TODO
- [ ] clean `geometry/internal_orientation.py`: choose one method (Molnar?) and remove the others.
- [ ] in `geometry.internal_orientation.py`, rename and/or move `objective_function`
- [x] implement lens distortion
- [ ] implement lens distortion inversion
- [ ] implement & test global objective function as in _Molnar et al._, for one image only.
- [ ] implement & test global objective function as in _Molnar et al._, for several images.
- [ ] split images and GCPs in smaller tiles to make georeferencing easier.