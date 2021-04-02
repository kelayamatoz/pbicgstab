APP=bicgstab_no_preconditioning

omp:
	g++ -I ../eigen/ -fopenmp ${APP}.cpp -o ${APP}_omp

no_omp:
	g++ -I ../eigen/ ${APP}.cpp -o ${APP}_no_omp

clean:
	rm -f ${APP}_omp ${APP}_no_omp
