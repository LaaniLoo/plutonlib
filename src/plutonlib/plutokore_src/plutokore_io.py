from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import object
import os as _os
import numpy as _np
import array as _array
from pathlib import Path

# check if we have h5py available
try:
    import h5py as h5

    hasH5 = True
except ImportError:
    hasH5 = False


def nlast_info(w_dir=None, datatype=None):
    """Prints the information of the last step of the simulation as obtained from out files

    **Inputs**:

      w_dir -- path to the directory which has the dbltout(or flt.out) and the data\n
      datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.

    **Outputs**:

      This function returns a dictionary with following keywords - \n

      nlast -- The ns for the last file saved.\n
      time -- The simulation time for the last file saved.\n
      dt -- The time step dt for the last file. \n
      Nstep -- The Nstep value for the last file saved.


    **Usage**:

      In case the data is 'float'.

      ``wdir = /path/to/data/directory``\n
      ``import pyPLUTO as pp``\n
      ``A = pp.nlast_info(w_dir=wdir,datatype='float')``
    """
    if w_dir is None:
        w_dir = curdir()
    # ensure trailing slash exists on directory path
    w_dir = _os.path.join(w_dir, "")
    if datatype == "float":
        fname_v = w_dir + "flt.out"
    elif datatype == "vtk":
        fname_v = w_dir + "vtk.out"
    else: #NOTE Try dbl or h5\
        if _os.path.isfile(_os.path.join(w_dir,r"dbl.h5.out")):
            fname_v = w_dir + "dbl.h5.out"
        elif _os.path.isfile(_os.path.join(w_dir,r"dbl.out")):
            fname_v = w_dir + "dbl.out"
        else:
            print("Neither dbl.h5.out or dbl.out files are found")


    with open(fname_v, "r") as f:
        last_line = f.readlines()[-1].split()
    nlast = int(last_line[0])
    sim_time = float(last_line[1])
    Dt = float(last_line[2])
    Nstep = int(last_line[3])

    return {"nlast": nlast, "time": sim_time, "dt": Dt, "Nstep": Nstep}


class pload(object):
    def __init__(
        self,
        ns,
        w_dir=None,
        datatype=None,
        level=0,
        x1range=None,
        x2range=None,
        x3range=None,
        mmap=False,
        load_data=True,
    ):
        """Loads the data.

        **Inputs**:

          ns -- Step Number of the data file\n
          w_dir -- path to the directory which has the data files\n
          datatype -- Datatype (default = 'double')

        **Outputs**:

          pyPLUTO pload object whose keys are arrays of data values.

        """
        self.NStep = ns
        self.Dt = 0.0

        self.n1 = 0
        self.n2 = 0
        self.n3 = 0

        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.dx1 = []
        self.dx2 = []
        self.dx3 = []

        self.x1range = x1range
        self.x2range = x2range
        self.x3range = x3range

        self.mmap = mmap
        self.load_data = load_data

        self.NStepStr = str(self.NStep)
        while len(self.NStepStr) < 4:
            self.NStepStr = "0" + self.NStepStr

        if datatype is None:
            datatype = "double"
        self.datatype = datatype

        if (not hasH5) and (datatype == "hdf5"):
            raise Exception("The h5py library is required to load .hdf5 files")
            return

        self.level = level

        if w_dir is None:
            w_dir = _os.getcwd() + "/"
        # make sure w_dir has a trailing path separator
        w_dir = _os.path.join(w_dir, "")
        self.wdir = w_dir

        Data_dictionary = self.ReadDataFile(self.NStepStr)
        for keys in Data_dictionary:
            object.__setattr__(self, keys, Data_dictionary.get(keys))

    def ReadTimeInfo(self, timefile):
        """Read time info from the outfiles.

        **Inputs**:

          timefile -- name of the out file which has timing information.

        """

        if self.datatype == "hdf5":
            fh5 = h5.File(timefile, "r")
            self.sim_time = fh5.attrs.get("time")
            # self.Dt = 1.e-2 # Should be erased later given the level in AMR
            fh5.close()
        else:
            ns = self.NStep
            with open(timefile, "r") as f_var:
                tlist = []
                for line in f_var.readlines():
                    tlist.append(line.split())
            self.sim_time = float(tlist[ns][1])
            self.Dt = float(tlist[ns][2])

    def ReadVarFile(self, varfile):
        """Read variable names from the outfiles.

        **Inputs**:

          varfile -- name of the out file which has variable information.

        """
        if self.datatype == "hdf5":
            fh5 = h5.File(varfile, "r")
            self.filetype = "single_file"
            self.endianess = ">"  # not used with AMR, kept for consistency
            self.vars = []
            for iv in range(fh5.attrs.get("num_components")):
                self.vars.append(fh5.attrs.get("component_" + str(iv)))
            fh5.close()
        else:
            vfp = open(varfile, "r")
            varinfo = vfp.readline().split()
            self.filetype = varinfo[4]
            self.endianess = varinfo[5]
            self.vars = varinfo[6:]
            vfp.close()

    def ReadGridFile(self, gridfile):
        """Read grid values from the grid.out file.

        **Inputs**:

          gridfile -- name of the grid.out file which has information about the grid.

        """
        xL = []
        xR = []
        nmax = []
        with open(gridfile, "r") as gfp:
            for i in gfp.readlines():
                if len(i.split()) == 1:
                    try:
                        int(i.split()[0])
                        nmax.append(int(i.split()[0]))
                    except:
                        pass

                if len(i.split()) == 3:
                    try:
                        int(i.split()[0])
                        xL.append(float(i.split()[1]))
                        xR.append(float(i.split()[2]))
                    except:
                        if i.split()[1] == "GEOMETRY:":
                            self.geometry = i.split()[2]
                        pass

        self.n1, self.n2, self.n3 = nmax
        n1 = self.n1
        n1p2 = self.n1 + self.n2
        n1p2p3 = self.n1 + self.n2 + self.n3
        self.x1 = _np.asarray([0.5 * (xL[i] + xR[i]) for i in range(n1)])
        self.dx1 = _np.asarray([(xR[i] - xL[i]) for i in range(n1)])
        self.x2 = _np.asarray([0.5 * (xL[i] + xR[i]) for i in range(n1, n1p2)])
        self.dx2 = _np.asarray([(xR[i] - xL[i]) for i in range(n1, n1p2)])
        self.x3 = _np.asarray([0.5 * (xL[i] + xR[i]) for i in range(n1p2, n1p2p3)])
        self.dx3 = _np.asarray([(xR[i] - xL[i]) for i in range(n1p2, n1p2p3)])

        # Stores the total number of points in '_tot' variable in case only
        # a portion of the domain is loaded. Redefine the x and dx arrays
        # to match the requested ranges
        self.n1_tot = self.n1
        self.n2_tot = self.n2
        self.n3_tot = self.n3
        if self.x1range != None:
            self.n1_tot = self.n1
            self.irange = list(
                range(
                    abs(self.x1 - self.x1range[0]).argmin(),
                    abs(self.x1 - self.x1range[1]).argmin() + 1,
                )
            )
            self.n1 = len(self.irange)
            self.x1 = self.x1[self.irange]
            self.dx1 = self.dx1[self.irange]
        else:
            self.irange = list(range(self.n1))
        if self.x2range != None:
            self.n2_tot = self.n2
            self.jrange = list(
                range(
                    abs(self.x2 - self.x2range[0]).argmin(),
                    abs(self.x2 - self.x2range[1]).argmin() + 1,
                )
            )
            self.n2 = len(self.jrange)
            self.x2 = self.x2[self.jrange]
            self.dx2 = self.dx2[self.jrange]
        else:
            self.jrange = list(range(self.n2))
        if self.x3range != None:
            self.n3_tot = self.n3
            self.krange = list(
                range(
                    abs(self.x3 - self.x3range[0]).argmin(),
                    abs(self.x3 - self.x3range[1]).argmin() + 1,
                )
            )
            self.n3 = len(self.krange)
            self.x3 = self.x3[self.krange]
            self.dx3 = self.dx3[self.krange]
        else:
            self.krange = list(range(self.n3))
        self.Slice = (
            (self.x1range != None) or (self.x2range != None) or (self.x3range != None)
        )

        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        self.x1r = _np.zeros(len(self.x1) + 1)
        self.x1r[1:] = self.x1 + self.dx1 / 2.0
        self.x1r[0] = self.x1r[1] - self.dx1[0]
        self.x2r = _np.zeros(len(self.x2) + 1)
        self.x2r[1:] = self.x2 + self.dx2 / 2.0
        self.x2r[0] = self.x2r[1] - self.dx2[0]
        self.x3r = _np.zeros(len(self.x3) + 1)
        self.x3r[1:] = self.x3 + self.dx3 / 2.0
        self.x3r[0] = self.x3r[1] - self.dx3[0]

        prodn = self.n1 * self.n2 * self.n3
        if prodn == self.n1:
            self.nshp = self.n1
        elif prodn == self.n1 * self.n2:
            self.nshp = (self.n2, self.n1)
        else:
            self.nshp = (self.n3, self.n2, self.n1)

    def DataScanVTK(self, fp, n1, n2, n3, endian, dtype):
        """Scans the VTK data files.

        **Inputs**:

         fp -- Data file pointer\n
         n1 -- No. of points in X1 direction\n
         n2 -- No. of points in X2 direction\n
         n3 -- No. of points in X3 direction\n
         endian -- Endianess of the data\n
         dtype -- datatype

        **Output**:

          Dictionary consisting of variable names as keys and its values.

        """
        ks = []
        vtkvar = []
        while True:
            l = fp.readline()
            try:
                l.split()[0]
            except IndexError:
                pass
            else:
                if l.split()[0] == "SCALARS":
                    ks.append(l.split()[1])
                elif l.split()[0] == "LOOKUP_TABLE":
                    A = _array.array(dtype)
                    fmt = endian + str(n1 * n2 * n3) + dtype
                    nb = _np.dtype(fmt).itemsize
                    A.fromstring(fp.read(nb))
                    if self.Slice:
                        darr = _np.zeros((n1 * n2 * n3))
                        indxx = _np.sort(
                            [
                                n3_tot * n2_tot * k + j * n2_tot + i
                                for i in self.irange
                                for j in self.jrange
                                for k in self.krange
                            ]
                        )
                        if sys.byteorder != self.endianess:
                            A.byteswap()
                        for ii, iii in enumerate(indxx):
                            darr[ii] = A[iii]
                        vtkvar_buf = [darr]
                    else:
                        vtkvar_buf = _np.frombuffer(A, dtype=_np.dtype(fmt))
                    vtkvar.append(_np.reshape(vtkvar_buf, self.nshp).transpose())
                else:
                    pass
            if l == "":
                break

        vtkvardict = dict(list(zip(ks, vtkvar)))
        return vtkvardict

    def DataScanHDF5(self, fp, myvars, ilev):
        """Scans the Chombo HDF5 data files for AMR in PLUTO.

        **Inputs**:

          fp     -- Data file pointer\n
          myvars -- Names of the variables to read\n
          ilev   -- required AMR level

        **Output**:

          Dictionary consisting of variable names as keys and its values.

        **Note**:

          Due to the particularity of AMR, the grid arrays loaded in ReadGridFile are overwritten here.

        """
        # Read the grid information
        dim = fp["Chombo_global"].attrs.get("SpaceDim")
        nlev = fp.attrs.get("num_levels")
        il = min(nlev - 1, ilev)
        lev = []
        for i in range(nlev):
            lev.append("level_" + str(i))
        freb = _np.zeros(nlev, dtype="int")
        for i in range(il + 1)[::-1]:
            fl = fp[lev[i]]
            if i == il:
                pdom = fl.attrs.get("prob_domain")
                dx = fl.attrs.get("dx")
                dt = fl.attrs.get("dt")
                ystr = 1.0
                zstr = 1.0
                logr = 0
                try:
                    geom = fl.attrs.get("geometry")
                    logr = fl.attrs.get("logr")
                    if dim == 2:
                        ystr = fl.attrs.get("g_x2stretch")
                    elif dim == 3:
                        zstr = fl.attrs.get("g_x3stretch")
                except:
                    print("Old HDF5 file, not reading stretch and logr factors")
                freb[i] = 1
                x1b = fl.attrs.get("domBeg1")
                if dim == 1:
                    x2b = 0
                else:
                    x2b = fl.attrs.get("domBeg2")
                if dim == 1 or dim == 2:
                    x3b = 0
                else:
                    x3b = fl.attrs.get("domBeg3")
                jbeg = 0
                jend = 0
                ny = 1
                kbeg = 0
                kend = 0
                nz = 1
                if dim == 1:
                    ibeg = pdom[0]
                    iend = pdom[1]
                    nx = iend - ibeg + 1
                elif dim == 2:
                    ibeg = pdom[0]
                    iend = pdom[2]
                    nx = iend - ibeg + 1
                    jbeg = pdom[1]
                    jend = pdom[3]
                    ny = jend - jbeg + 1
                elif dim == 3:
                    ibeg = pdom[0]
                    iend = pdom[3]
                    nx = iend - ibeg + 1
                    jbeg = pdom[1]
                    jend = pdom[4]
                    ny = jend - jbeg + 1
                    kbeg = pdom[2]
                    kend = pdom[5]
                    nz = kend - kbeg + 1
            else:
                rat = fl.attrs.get("ref_ratio")
                freb[i] = rat * freb[i + 1]

        dx0 = dx * freb[0]

        ## Allow to load only a portion of the domain
        if self.x1range != None:
            if logr == 0:
                self.x1range = self.x1range - x1b
            else:
                self.x1range = [log(self.x1range[0] / x1b), log(self.x1range[1] / x1b)]
            ibeg0 = min(self.x1range) / dx0
            iend0 = max(self.x1range) / dx0
            ibeg = max([ibeg, int(ibeg0 * freb[0])])
            iend = min([iend, int(iend0 * freb[0] - 1)])
            nx = iend - ibeg + 1
        if self.x2range != None:
            self.x2range = (self.x2range - x2b) / ystr
            jbeg0 = min(self.x2range) / dx0
            jend0 = max(self.x2range) / dx0
            jbeg = max([jbeg, int(jbeg0 * freb[0])])
            jend = min([jend, int(jend0 * freb[0] - 1)])
            ny = jend - jbeg + 1
        if self.x3range != None:
            self.x3range = (self.x3range - x3b) / zstr
            kbeg0 = min(self.x3range) / dx0
            kend0 = max(self.x3range) / dx0
            kbeg = max([kbeg, int(kbeg0 * freb[0])])
            kend = min([kend, int(kend0 * freb[0] - 1)])
            nz = kend - kbeg + 1

        ## Create uniform grids at the required level
        if logr == 0:
            x1 = x1b + (ibeg + _np.array(list(range(nx))) + 0.5) * dx
        else:
            x1 = (
                x1b
                * (
                    exp((ibeg + _np.array(list(range(nx))) + 1) * dx)
                    + exp((ibeg + _np.array(list(range(nx)))) * dx)
                )
                * 0.5
            )

        x2 = x2b + (jbeg + _np.array(list(range(ny))) + 0.5) * dx * ystr
        x3 = x3b + (kbeg + _np.array(list(range(nz))) + 0.5) * dx * zstr
        if logr == 0:
            dx1 = _np.ones(nx) * dx
        else:
            dx1 = x1b * (
                exp((ibeg + _np.array(list(range(nx))) + 1) * dx)
                - exp((ibeg + _np.array(list(range(nx)))) * dx)
            )
        dx2 = _np.ones(ny) * dx * ystr
        dx3 = _np.ones(nz) * dx * zstr

        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        x1r = _np.zeros(len(x1) + 1)
        x1r[1:] = x1 + dx1 / 2.0
        x1r[0] = x1r[1] - dx1[0]
        x2r = _np.zeros(len(x2) + 1)
        x2r[1:] = x2 + dx2 / 2.0
        x2r[0] = x2r[1] - dx2[0]
        x3r = _np.zeros(len(x3) + 1)
        x3r[1:] = x3 + dx3 / 2.0
        x3r[0] = x3r[1] - dx3[0]
        NewGridDict = dict(
            [
                ("n1", nx),
                ("n2", ny),
                ("n3", nz),
                ("x1", x1),
                ("x2", x2),
                ("x3", x3),
                ("x1r", x1r),
                ("x2r", x2r),
                ("x3r", x3r),
                ("dx1", dx1),
                ("dx2", dx2),
                ("dx3", dx3),
                ("Dt", dt),
            ]
        )

        # Variables table
        nvar = len(myvars)
        vars = _np.zeros((nx, ny, nz, nvar))

        LevelDic = {
            "nbox": 0,
            "ibeg": ibeg,
            "iend": iend,
            "jbeg": jbeg,
            "jend": jend,
            "kbeg": kbeg,
            "kend": kend,
        }
        AMRLevel = []
        AMRBoxes = _np.zeros((nx, ny, nz))
        for i in range(il + 1):
            AMRLevel.append(LevelDic.copy())
            fl = fp[lev[i]]
            data = fl["data:datatype=0"]
            boxes = fl["boxes"]
            nbox = len(boxes["lo_i"])
            AMRLevel[i]["nbox"] = nbox
            ncount = 0
            AMRLevel[i]["box"] = []
            for j in range(nbox):  # loop on all boxes of a given level
                AMRLevel[i]["box"].append(
                    {
                        "x0": 0.0,
                        "x1": 0.0,
                        "ib": 0,
                        "ie": 0,
                        "y0": 0.0,
                        "y1": 0.0,
                        "jb": 0,
                        "je": 0,
                        "z0": 0.0,
                        "z1": 0.0,
                        "kb": 0,
                        "ke": 0,
                    }
                )
                # Box indexes
                ib = boxes[j]["lo_i"]
                ie = boxes[j]["hi_i"]
                nbx = ie - ib + 1
                jb = 0
                je = 0
                nby = 1
                kb = 0
                ke = 0
                nbz = 1
                if dim > 1:
                    jb = boxes[j]["lo_j"]
                    je = boxes[j]["hi_j"]
                    nby = je - jb + 1
                if dim > 2:
                    kb = boxes[j]["lo_k"]
                    ke = boxes[j]["hi_k"]
                    nbz = ke - kb + 1
                szb = nbx * nby * nbz * nvar
                # Rescale to current level
                kb = kb * freb[i]
                ke = (ke + 1) * freb[i] - 1
                jb = jb * freb[i]
                je = (je + 1) * freb[i] - 1
                ib = ib * freb[i]
                ie = (ie + 1) * freb[i] - 1

                # Skip boxes lying outside ranges
                if (
                    (ib > iend)
                    or (ie < ibeg)
                    or (jb > jend)
                    or (je < jbeg)
                    or (kb > kend)
                    or (ke < kbeg)
                ):
                    ncount = ncount + szb
                else:

                    ### Read data
                    q = data[ncount : ncount + szb].reshape((nvar, nbz, nby, nbx)).T

                    ### Find boxes intersections with current domain ranges
                    ib0 = max([ibeg, ib])
                    ie0 = min([iend, ie])
                    jb0 = max([jbeg, jb])
                    je0 = min([jend, je])
                    kb0 = max([kbeg, kb])
                    ke0 = min([kend, ke])

                    ### Store box corners in the AMRLevel structure
                    if logr == 0:
                        AMRLevel[i]["box"][j]["x0"] = x1b + dx * (ib0)
                        AMRLevel[i]["box"][j]["x1"] = x1b + dx * (ie0 + 1)
                    else:
                        AMRLevel[i]["box"][j]["x0"] = x1b * exp(dx * (ib0))
                        AMRLevel[i]["box"][j]["x1"] = x1b * exp(dx * (ie0 + 1))
                    AMRLevel[i]["box"][j]["y0"] = x2b + dx * (jb0) * ystr
                    AMRLevel[i]["box"][j]["y1"] = x2b + dx * (je0 + 1) * ystr
                    AMRLevel[i]["box"][j]["z0"] = x3b + dx * (kb0) * zstr
                    AMRLevel[i]["box"][j]["z1"] = x3b + dx * (ke0 + 1) * zstr
                    AMRLevel[i]["box"][j]["ib"] = ib0
                    AMRLevel[i]["box"][j]["ie"] = ie0
                    AMRLevel[i]["box"][j]["jb"] = jb0
                    AMRLevel[i]["box"][j]["je"] = je0
                    AMRLevel[i]["box"][j]["kb"] = kb0
                    AMRLevel[i]["box"][j]["ke"] = ke0
                    AMRBoxes[
                        ib0 - ibeg : ie0 - ibeg + 1,
                        jb0 - jbeg : je0 - jbeg + 1,
                        kb0 - kbeg : ke0 - kbeg + 1,
                    ] = il

                    ### Extract the box intersection from data stored in q
                    cib0 = (ib0 - ib) / freb[i]
                    cie0 = (ie0 - ib) / freb[i]
                    cjb0 = (jb0 - jb) / freb[i]
                    cje0 = (je0 - jb) / freb[i]
                    ckb0 = (kb0 - kb) / freb[i]
                    cke0 = (ke0 - kb) / freb[i]
                    q1 = _np.zeros(
                        (cie0 - cib0 + 1, cje0 - cjb0 + 1, cke0 - ckb0 + 1, nvar)
                    )
                    q1 = q[cib0 : cie0 + 1, cjb0 : cje0 + 1, ckb0 : cke0 + 1, :]

                    # Remap the extracted portion
                    if dim == 1:
                        new_shape = (ie0 - ib0 + 1, 1)
                    elif dim == 2:
                        new_shape = (ie0 - ib0 + 1, je0 - jb0 + 1)
                    else:
                        new_shape = (ie0 - ib0 + 1, je0 - jb0 + 1, ke0 - kb0 + 1)

                    stmp = list(new_shape)
                    while stmp.count(1) > 0:
                        stmp.remove(1)
                    new_shape = tuple(stmp)

                    myT = Tools()
                    for iv in range(nvar):
                        vars[
                            ib0 - ibeg : ie0 - ibeg + 1,
                            jb0 - jbeg : je0 - jbeg + 1,
                            kb0 - kbeg : ke0 - kbeg + 1,
                            iv,
                        ] = myT.congrid(
                            q1[:, :, :, iv].squeeze(),
                            new_shape,
                            method="linear",
                            minusone=True,
                        ).reshape(
                            (ie0 - ib0 + 1, je0 - jb0 + 1, ke0 - kb0 + 1)
                        )
                    ncount = ncount + szb

        h5vardict = {}
        for iv in range(nvar):
            h5vardict[myvars[iv]] = vars[:, :, :, iv].squeeze()
        AMRdict = dict([("AMRBoxes", AMRBoxes), ("AMRLevel", AMRLevel)])
        OutDict = dict(NewGridDict)
        OutDict.update(AMRdict)
        OutDict.update(h5vardict)
        return OutDict

    def DataScan(self, fp, n1, n2, n3, endian, dtype, var_index, off=None):
        """Scans the data files in all formats.

        **Inputs**:

          fp -- Data file pointer\n
          n1 -- No. of points in X1 direction\n
          n2 -- No. of points in X2 direction\n
          n3 -- No. of points in X3 direction\n
          endian -- Endianess of the data\n
          dtype -- datatype, eg : double, float, vtk, hdf5\n
          off -- offset (for avoiding staggered B fields)

        **Output**:

          Dictionary consisting of variable names as keys and its values.

        """
        if off is not None:
            off_fmt = endian + str(off) + dtype
            nboff = _np.dtype(off_fmt).itemsize
            fp.read(nboff)

        n1_tot = self.n1_tot
        n2_tot = self.n2_tot
        n3_tot = self.n3_tot

        # A = _array.array(dtype)
        # fmt = endian + str(n1_tot * n2_tot * n3_tot) + dtype
        # print(fmt)
        # nb = _np.dtype(fmt).itemsize
        # A.fromstring(fp.read(nb))
        # A = _np.fromstring(fp.read(n1_tot*n2_tot*n3_tot*8), dtype=_np.dtype(endian + dtype))
        dt = _np.dtype(endian + dtype)
        if self.mmap:
            return _np.memmap(
                fp,
                dtype=dt,
                mode="c",
                offset=n1_tot * n2_tot * n3_tot * dt.itemsize * var_index,
                shape=self.nshp,
                order="C",
            ).T
        else:
            A = _np.fromfile(fp, dtype=dt, sep="", count=n1_tot * n2_tot * n3_tot)

        if self.Slice:
            darr = _np.zeros((n1 * n2 * n3))
            indxx = _np.sort(
                [
                    n3_tot * n2_tot * k + j * n2_tot + i
                    for i in self.irange
                    for j in self.jrange
                    for k in self.krange
                ]
            )
            if sys.byteorder != self.endianess:
                A.byteswap(true)
            for ii, iii in enumerate(indxx):
                darr[ii] = A[iii]
            darr = [darr]
        else:
            darr = A
            # darr = _np.frombuffer(A, dtype=_np.dtype(fmt))

        return _np.reshape(darr, self.nshp).transpose()

    def ReadSingleFile(self, datafilename, myvars, n1, n2, n3, endian, dtype, ddict):
        """
        Reads a single data file, data.****.dtype.

        Parameters
        ----------
        datafilename
            Data file name

        n1
            No. of points in X1 direction

        n2
            No. of points in X2 direction

        n3
            No. of points in X3 direction

        endian
            Endianess of the data

        dtype
            datatype

        ddict
            Dictionary containing the Grid and Time information,
            which is updated.

        Returns
        -------
        Updated dictionary consisting of variable names as keys and its values.
        """
        if self.datatype == "hdf5":
            fp = h5.File(datafilename, "r")
        elif self.mmap:
            fp = datafilename
        else:
            fp = open(datafilename, "rb")

        if self.datatype == "vtk":
            vtkd = self.DataScanVTK(fp, n1, n2, n3, endian, dtype)
            ddict.update(vtkd)
        elif self.datatype == "hdf5":
            h5d = self.DataScanHDF5(fp, myvars, self.level)
            ddict.update(h5d)
        else:
            for i in range(len(myvars)):
                if myvars[i] == "bx1s":
                    ddict.update(
                        {
                            myvars[i]: self.DataScan(
                                fp, n1, n2, n3, endian, dtype, var_index=i, off=n2 * n3
                            )
                        }
                    )
                elif myvars[i] == "bx2s":
                    ddict.update(
                        {
                            myvars[i]: self.DataScan(
                                fp, n1, n2, n3, endian, dtype, var_index=i, off=n3 * n1
                            )
                        }
                    )
                elif myvars[i] == "bx3s":
                    ddict.update(
                        {
                            myvars[i]: self.DataScan(
                                fp, n1, n2, n3, endian, dtype, var_index=i, off=n1 * n2
                            )
                        }
                    )
                else:
                    ddict.update(
                        {
                            myvars[i]: self.DataScan(
                                fp, n1, n2, n3, endian, dtype, var_index=i
                            )
                        }
                    )

        if not self.mmap:
            fp.close()

    def ReadMultipleFiles(
        self, nstr, dataext, myvars, n1, n2, n3, endian, dtype, ddict
    ):
        """Reads a  multiple data files, varname.****.dataext.

        **Inputs**:

          nstr -- File number in form of a string\n
          dataext -- Data type of the file, e.g., 'dbl', 'flt' or 'vtk' \n
          myvars -- List of variable names to be read\n
          n1 -- No. of points in X1 direction\n
          n2 -- No. of points in X2 direction\n
          n3 -- No. of points in X3 direction\n
          endian -- Endianess of the data\n
          dtype -- datatype\n
          ddict -- Dictionary containing Grid and Time Information
          which is updated.

        **Output**:

          Updated Dictionary consisting of variable names as keys and its values.

        """
        for i in range(len(myvars)):
            datafilename = self.wdir + myvars[i] + "." + nstr + dataext
            if self.mmap:
                fp = datafilename
            else:
                fp = open(datafilename, "rb")
            if self.datatype == "vtk":
                ddict.update(self.DataScanVTK(fp, n1, n2, n3, endian, dtype))
            else:
                ddict.update(
                    {
                        myvars[i]: self.DataScan(
                            fp, n1, n2, n3, endian, dtype, var_index=0
                        )
                    }
                )
            if not self.mmap:
                fp.close()

    def ReadDataFile(self, num):
        """Reads the data file generated from PLUTO code.

        **Inputs**:

          num -- Data file number in form of an Integer.

        **Outputs**:

          Dictionary that contains all information about Grid, Time and
          variables.

        """
        gridfile = self.wdir + "grid.out"
        if self.datatype == "float":
            dtype = "f"
            varfile = self.wdir + "flt.out"
            dataext = ".flt"
        elif self.datatype == "vtk":
            dtype = "f"
            varfile = self.wdir + "vtk.out"
            dataext = ".vtk"
        elif self.datatype == "hdf5":
            dtype = "d"
            dataext = ".hdf5"
            nstr = num
            varfile = self.wdir + "data." + nstr + dataext
        elif self.datatype == "png":
            dataext = ".png"
            varfile = self.wdir + "png.out"
        else:
            dtype = "d"
            if _os.path.isfile(_os.path.join(self.wdir,r"dbl.h5.out")):
                varfile = self.wdir + "dbl.h5.out"
                dataext = ".dbl.h5"
            elif _os.path.isfile(_os.path.join(self.wdir,r"dbl.out")):
                varfile = self.wdir + "dbl.out"
                dataext = ".dbl"


        self.ReadVarFile(varfile)
        self.ReadGridFile(gridfile)
        self.ReadTimeInfo(varfile)
        nstr = num
        if self.endianess == "big":
            endian = ">"
        elif self.datatype == "vtk":
            endian = ">"
        else:
            endian = "<"

        D = [
            ("NStep", self.NStep),
            ("sim_time", self.sim_time),
            ("Dt", self.Dt),
            ("n1", self.n1),
            ("n2", self.n2),
            ("n3", self.n3),
            ("x1", self.x1),
            ("x2", self.x2),
            ("x3", self.x3),
            ("dx1", self.dx1),
            ("dx2", self.dx2),
            ("dx3", self.dx3),
            ("endianess", self.endianess),
            ("datatype", self.datatype),
            ("filetype", self.filetype),
        ]
        ddict = dict(D)

        if self.load_data:
            if self.filetype == "single_file":
                datafilename = self.wdir + "data." + nstr + dataext
                self.ReadSingleFile(
                    datafilename,
                    self.vars,
                    self.n1,
                    self.n2,
                    self.n3,
                    endian,
                    dtype,
                    ddict,
                )
            elif self.filetype == "multiple_files":
                self.ReadMultipleFiles(
                    nstr,
                    dataext,
                    self.vars,
                    self.n1,
                    self.n2,
                    self.n3,
                    endian,
                    dtype,
                    ddict,
                )
            else:
                raise Exception("Wrong file type {0}, check pluto.ini for filetype.")

        return ddict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ploadparticles(object):
    def __init__(self, ns, wdir, datatype=None):
        self.ns = ns
        self.wdir = Path(wdir)

        if datatype is None:
            datatype = "double"

        if datatype == "double":
            self.ext = "dbl"
        elif datatype == "float":
            self.ext = "flt"
        elif datatype == "vtk":
            self.ext = "vtk"

        if self.ext == "dbl":
            self.extsize = 8
        if self.ext == "flt":
            self.extsize = 4
        if self.ext == "vtk":
            self.extsize = 4

        self.Header = {}
        try:
            self.GetParticleDataFileName()
        except (IOError):
            print("!ERROR! - Particle Data file not found : %s" % self.fname)
        else:
            if self.ext == "vtk":
                Part_Data_dict = self.GetParticleDataVTK()
            else:
                self.ReadParticleFile()
                self.Header["time"] = float(self.Header["time"])
                Part_Data_dict = self.GetParticleData()

            self.partdata_keys = []
            for keys in Part_Data_dict:
                self.partdata_keys.append(keys)
                object.__setattr__(self, keys, Part_Data_dict.get(keys))

    def GetParticleDataFileName(self):
        nsstr_ = str(self.ns)
        while len(nsstr_) < 4:
            nsstr_ = "0" + nsstr_
        self.fname = self.wdir / f"particles.{nsstr_}.{self.ext}"

    def GetParticleFileHeader(self, line_):
        if line_.split()[1] != "PLUTO":
            hlist = line_.split()[1:]
            if hlist[0] == "field_names":
                vars_ = hlist[1:]
                t = tuple([hlist[0], vars_])
                self.Header.update(dict([t]))
            elif hlist[0] == "field_dim":
                varsdim_ = [int(fd) for fd in hlist[1:]]
                t = tuple([hlist[0], varsdim_])
                self.Header.update(dict([t]))
            elif hlist[0] == "shk_thresh":
                shk_thresh_list = [float(st) for st in hlist[1:]]
                t = tuple([hlist[0], shk_thresh_list])
                self.Header.update(dict([t]))
            else:
                t = tuple(["".join(hlist[:-1]), hlist[-1]])
                self.Header.update(dict([t]))

    def ReadParticleFile(self):
        # print("Reading Particle Data file : %s"%self.fname)
        with open(self.fname, "rb") as fp_:
            for l in fp_.readlines():
                try:
                    ld = l.decode("ascii").strip()
                    if ld.split()[0] == "#":
                        self.GetParticleFileHeader(ld)
                    else:
                        break
                except UnicodeDecodeError:
                    break

        self.tot_fdim = _np.sum(_np.array(self.Header["field_dim"], dtype=_np.int))
        HeadBytes_ = int(self.Header["nparticles"]) * self.tot_fdim * self.extsize
        with open(self.fname, "rb") as fp_:
            scrh_ = fp_.read()
            self.datastr = scrh_[len(scrh_) - HeadBytes_ :]

    def GetParticleDataVTK(self):
        print("Reading Particle Data file : %s" % self.fname)
        fp_ = open(self.fname, "rb")
        ks_ = []
        vtk_scalars_ = []
        while True:
            l = fp_.readline()
            try:
                l.split()[0]
            except IndexError:
                pass
            else:
                if l.split()[0] == "POINTS":
                    np_ = int(l.split()[1])
                    endianess = ">"
                    dtyp_ = _np.dtype(endianess + "%df" % (3 * np_))
                    nb_ = _np.dtype(dtyp_).itemsize
                    posbuf_ = array.array("f")
                    posbuf_.fromstring(fp_.read(nb_))
                    pos_ = _np.frombuffer(posbuf_, dtype=_np.dtype(dtyp_))

                    pos1_ = _np.zeros(np_)
                    pos2_ = _np.zeros(np_)
                    pos3_ = _np.zeros(np_)
                    for i in range(np_):
                        pos1_[i] = pos_[0][3 * i]
                        pos2_[i] = pos_[0][3 * i + 1]
                        pos3_[i] = pos_[0][3 * i + 2]

                elif l.split()[0] == "VECTORS":
                    endianess = ">"
                    dtyp_ = _np.dtype(endianess + "%df" % (3 * np_))
                    nb_ = _np.dtype(dtyp_).itemsize
                    velbuf_ = array.array("f")
                    velbuf_.fromstring(fp_.read(nb_))
                    vel_ = _np.frombuffer(velbuf_, dtype=_np.dtype(dtyp_))

                    vel1_ = _np.zeros(np_)
                    vel2_ = _np.zeros(np_)
                    vel3_ = _np.zeros(np_)
                    for i in range(np_):
                        vel1_[i] = vel_[0][3 * i]
                        vel2_[i] = vel_[0][3 * i + 1]
                        vel3_[i] = vel_[0][3 * i + 2]

                elif l.split()[0] == "SCALARS":
                    ks_.append(l.split()[1])
                elif l.split()[0] == "LOOKUP_TABLE":
                    endianess = ">"
                    if ks_[-1] == "Identity":
                        dtyp_ = _np.dtype(endianess + "%di" % np_)
                    else:
                        dtyp_ = _np.dtype(endianess + "%df" % np_)

                    nb_ = _np.dtype(dtyp_).itemsize
                    scalbuf_ = array.array("f")
                    scalbuf_.fromstring(fp_.read(nb_))
                    scal_ = _np.frombuffer(scalbuf_, dtype=_np.dtype(dtyp_))

                    vtk_scalars_.append(scal_[0])
                else:
                    pass

            if l == "":
                break
        ks_ = ["id", "color", "tinj"]
        vtkvardict = dict(zip(ks_, vtk_scalars_))
        tup_ = [
            ("x1", pos1_),
            ("x2", pos2_),
            ("x3", pos3_),
            ("vx1", vel1_),
            ("vx2", vel2_),
            ("vx3", vel3_),
        ]
        vtkvardict.update(dict(tup_))
        return vtkvardict

    def GetParticleData(self):
        vars_ = self.Header["field_names"]
        endianess = "<"
        if self.Header["endianity"] == "big":
            endianess = ">"
        dtyp_ = _np.dtype(endianess + self.ext[0])
        DataDict_ = self.Header
        np_ = int(DataDict_["nparticles"])
        data_ = _np.fromstring(self.datastr, dtype=dtyp_)

        fdims = _np.array(self.Header["field_dim"], dtype=_np.int)
        indx = _np.where(fdims == 1)[0]
        spl_cnt = len(indx)
        counter = 0

        if np_ <= 0:
            return DataDict_

        reshaped_data = data_.reshape(np_, self.tot_fdim)
        tup_ = []

        ind = 0
        for c, v in enumerate(vars_):
            v_data = reshaped_data[:, ind : ind + fdims[c]]
            if fdims[c] == 1:
                v_data = v_data.reshape((np_))
            tup_.append((v, v_data))
            ind += fdims[c]
        DataDict_.update(dict(tup_))
        return DataDict_
