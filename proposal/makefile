all: pdf

pdf: woa_proposal.md ref.bib
	pandoc -F pandoc-crossref --citeproc woa_proposal.md -o woa_proposal.pdf

clean:
	rm woa_proposal.pdf