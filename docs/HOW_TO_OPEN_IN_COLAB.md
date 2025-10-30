# How to Open Your Notebook in Colab

## 🎯 Three Ways to Access

### Method 1: Click the Badge (Easiest)
1. Go to your GitHub repository: https://github.com/marioknicola/synthsup-speechMRI-recon
2. Click the "Open in Colab" badge at the top of the README
3. The notebook opens directly in Colab - ready to run!

### Method 2: Direct URL
Open this link in your browser:
```
https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb
```

### Method 3: From Google Colab
1. Go to https://colab.research.google.com/
2. Click "GitHub" tab
3. Enter: `marioknicola/synthsup-speechMRI-recon`
4. Select `colab_cross_validation.ipynb`

---

## ✅ What You Get

When you click the badge or URL, Colab will:
- ✅ Automatically open your notebook
- ✅ Connect to a free runtime (CPU by default)
- ✅ Show all your cells ready to run
- ✅ Allow you to enable GPU (Runtime → Change runtime type → GPU)

---

## 🚀 Quick Start Workflow

1. **Open notebook:** Click badge in README
2. **Enable GPU:** Runtime → Change runtime type → GPU
3. **Run cells:** Cell 1 (GPU check) → Cell 2 (clone repo)
4. **Upload data:** Cell 3 will prompt for ZIP files
5. **Configure:** Cell 4 & 5 (set subjects)
6. **Train:** Cell 6 starts training (12-18 hours)
7. **Download:** Cell 9 downloads results when done

---

## 📝 Important Notes

### Before Running:
- Prepare ZIP files locally first:
  ```bash
  zip -r Synth_LR_nii.zip Synth_LR_nii/
  zip -r HR_nii.zip HR_nii/
  ```

### No Google Drive Needed:
- Old workflow: Had to mount Drive
- New workflow: Just upload ZIPs, download results
- Simpler and faster!

### Saving Your Work:
- Notebook changes: File → Save a copy in Drive (if you modify cells)
- Training results: Download from Cell 9 after training
- Session is temporary: Download results before closing!

---

## 🔗 Share with Others

Send this link to anyone:
```
https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb
```

They can:
- Run your notebook with their own data
- No installation needed
- Free GPU access
- Perfect for collaborators or reviewers!

---

## 🎓 For Your Abstract/Paper

You can mention:
> "Training code and interactive notebook available at:  
> https://github.com/marioknicola/synthsup-speechMRI-recon  
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)"

Reviewers can literally click and run your experiments!

---

## ✨ That's It!

**Your notebook is now:**
- ✅ Accessible with one click
- ✅ No Drive dependencies
- ✅ GPU-ready
- ✅ Reproducible
- ✅ Shareable

Just click the badge and start training! 🚀
