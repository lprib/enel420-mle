%: %.ipynb
	jupyter nbconvert $< --to slides \
	--reveal-prefix "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.1.0" \
	--post serve
