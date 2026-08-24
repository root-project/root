# This was forked in 1993 and slightly adapted for ROOT. See
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/inc/x3d.h
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/src/x3d.c
# vs http://hpux.connect.org.uk/hppd/hpux/X11/Viewers/x3d-2.2/ download source and rename from .tar.gz to just .tar

target_sources(X3d PRIVATE x3d.h x3d.c)
