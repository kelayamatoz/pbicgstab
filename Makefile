APP=bicgstab_no_preconditioning

omp:
	# g++ -I ../eigen/ -march=native -fopenmp ${APP}.cpp -o ${APP}_omp
	g++ -I ../eigen/ -march=native -fopenmp ${APP}.cpp -o ${APP}_omp

no_omp:
	g++ -I ../eigen/ -march=native ${APP}.cpp -o ${APP}_no_omp

clean:
	rm -f ${APP}_omp ${APP}_no_omp
