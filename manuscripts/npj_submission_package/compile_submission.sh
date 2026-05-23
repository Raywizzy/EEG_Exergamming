#!/bin/bash
# NPJ Digital Medicine Submission Compilation Script
# Generated: 2025-09-20 10:12:55

set -e  # Exit on any error

echo "🚀 Compiling NPJ Digital Medicine Submission Package"
echo "=================================================="

# Navigate to package directory
cd "/Users/user/Desktop/EEG_Exergamming/manuscripts/npj_submission_package"

# Compile main manuscript
echo "📖 Compiling main manuscript..."
pdflatex manuscript_main.tex
bibtex manuscript_main
pdflatex manuscript_main.tex
pdflatex manuscript_main.tex

# Compile cover letter
echo "✉️ Compiling cover letter..."
pdflatex cover_letter.tex

# Check outputs
echo "✅ Checking compilation outputs..."
if [ -f "manuscript_main.pdf" ]; then
    echo "   ✓ Main manuscript PDF generated"
else
    echo "   ❌ Main manuscript PDF missing"
    exit 1
fi

if [ -f "cover_letter.pdf" ]; then
    echo "   ✓ Cover letter PDF generated"
else
    echo "   ❌ Cover letter PDF missing"
    exit 1
fi

# Count figures
figure_count=$(ls figures/*.png 2>/dev/null | wc -l)
echo "   ✓ Figures available: $figure_count"

# Final summary
echo ""
echo "🎉 SUBMISSION PACKAGE READY!"
echo "================================"
echo "Main manuscript: manuscript_main.pdf"
echo "Cover letter: cover_letter.pdf"
echo "Figures: $figure_count files in figures/"
echo "Total package size: $(du -sh . | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Review both PDFs carefully"
echo "2. Upload to Nature Portfolio submission system"
echo "3. Begin Phase V optimization in parallel"
echo ""
echo "Target journal: npj Digital Medicine"
echo "Expected review time: 8-12 weeks"
