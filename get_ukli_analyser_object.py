import ROOT
f1 = ROOT.TFile("skli_warwick_opticlib_analyser_v1.0.root")
obj1 = f1.Get('b5/collimator/b5_collimator_theta_air')
obj1.Print()
obj1.Draw()
