# KH5-ARGON images ortho-correction and georeferencing

## TODO
- [ ] clean `geometry/internal_orientation.py` orientation: choose one method (Molnar?) and remove the others. Move `objective_function`to `utils/optim.py`
- [ ] implement lens distortion
- [ ] implement & test global objective function as in _Molnar et al._, for one image only.
- [ ] implement & test global objective function as in _Molnar et al._, for several images.
- [ ] split images and GCPs in smaller tiles to make georeferencing easier.