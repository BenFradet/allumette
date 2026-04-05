set ylabel "time (s)"

set xrange [0:6]
set yrange [0:250]

plot "profile_plot.dat" using 2:1 with lines
