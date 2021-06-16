all: bevan_report.pdf

bevan_report.pdf: bevan_report.dvi
	dvipdf bevan_report.dvi

bevan_report.dvi: bevan_report.tex
	latex bevan_report.tex
	bibtex bevan_report
	latex bevan_report.tex
	latex bevan_report.tex

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.log
