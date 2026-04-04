set logscale x 10
set logscale y 10

plot "benchmark_plot.dat" using 1:2 title 'par' with lines,\
    "benchmark_plot.dat" using 1:3 title 'igp' with lines,\
    "benchmark_plot.dat" using 1:4 title 'gpu' with lines,\
    "benchmark_plot.dat" using 1:5 title 'seq' with lines
