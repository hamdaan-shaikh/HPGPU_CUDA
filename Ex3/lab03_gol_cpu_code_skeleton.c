#include <stdio.h>
#include <stdlib.h>
 
#define SRAND_VALUE 1985

void GOL_CPU(int dim, int *world, int *newWorld)
{
  int i, j, id;
  
  //Ghost row
  for(i = 1; i <= dim; i++) {
    //TODO: Copy first real row to bottom ghost row

    //TODO: Copy last real row to top ghost row

  }
  
  //Ghost column
  for(i = 0; i <= dim+1; i++) {
    //TODO: Copy first real column to right most ghost column

    //TODO: Copy last real column to left most ghost column
    
  }
  
  for (i = 1; i <= dim; i++) {
    for (j = 1; j <= dim; j++) {
    
      id = 0;//TODO:calculate matrix-position
      
      //world point
      int cell = world[id];
      int numNeighbors;

      // Get the number of neighbors for a world point
      numNeighbors = 	world[id] 	//TODO: lower
			+ world[id] 	//TODO: upper
			+ world[id]	//TODO: right
			+ world[id]	//TODO: left

			+ world[id]	//TODO: diagonal lower right
			+ world[id]	//TODO: diagonal upper right
			+ world[id]	//TODO: diagonal lower left
			+ world[id];	//TODO: diagonal upper left

      // game rules for Conways 23/3-world
      // 1) Any live cell with fewer than two live neighbours dies
      if (0)//TODO
	newWorld[id] = 0;

      // 2) Any live cell with two or three live neighbours lives
      else if (0)//TODO
	newWorld[id] = 1;

      // 3) Any live cell with more than three live neighbours dies
      else if (0)//TODO
	newWorld[id] = 0;

      // 4) Any dead cell with exactly three live neighbours becomes a live cell
      else if (0)//TODO
	newWorld[id] = 1;

      else
	newWorld[id] = cell;
    }
  }

}

void initRandom(int dim, int* world)
{
  
  int i, j;
 
  // Assign initial population randomly
  srand(SRAND_VALUE);
  for(i = 1; i<=dim; i++) {
    for(j = 1; j<=dim; j++) {
      //TODO
    }
  }
}


void create_PPM(int dim, int* world, FILE* fp)
{
  // Write header for ppm file (portable pixmap format)
  fprintf(fp, "P3\n");				// Portable Pixmap in ASCII encoding
  fprintf(fp, "%i %i\n", dim, dim);		// Dimension of the picture in pixel
  fprintf(fp, "255\n");				// Maximal color value
  
  // Sum cells and write world to file
  int total = 0;
  int i, j;
  
  for (i = 1; i<=dim; i++) {
    for (j = 1; j<=dim; j++) {
      if(0) { //TODO:is white?
	fprintf(fp, "255 255 255   ");		//dead cell is white
      } else {
	fprintf(fp, "  0   0    0   ");	//living cell is black
      }
      //calculate total cells alive
      //TODO
    }
    fprintf(fp, "\n");				// new column new line 
  }

  printf("Total Alive: %d\n", total);
}

int main(int argc, char* argv[])
{
  int iter;
  int* h_world;  //World on host
  int* tmpWorld; //tmp world pointer used to switch worlds

  FILE* fp;

  
  if (argc != 4) {
    fprintf(stderr, "usage: gameoflife <world dimension> <game steps> <output ppm file>\n");

    exit(1);
  }

  int dim = atoi(argv[1]); //Linear dimension of our world - without counting ghost cells
  int maxIter = atoi(argv[2]); //Number of game steps
  
  if ((fp = fopen(argv[3], "w")) == NULL) {
    fprintf(stderr, "can't create %s\n", argv[3]);
    exit(1);
  }

  size_t worldBytes = sizeof(int)*(dim+2)*(dim+2);

  // Allocate host World
  h_world =  malloc(worldBytes);
  
  //create initial world
  initRandom(dim, h_world);
  
  int* h_newWorld = malloc(worldBytes);

  // --- Main loop ---
  for (iter = 0; iter<maxIter; iter++) {
    GOL_CPU(dim, h_world, h_newWorld);
    
    // Swap worlds
    //TODO
  }
    
  create_PPM(dim, h_world, fp);

  fclose(fp);

  return 0;
}
