/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader;
import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.Scanner; // Import the Scanner class to read text files
import java.util.ArrayList;

/**
 *
 * @author Magooda
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    static ArrayList<ArrayList> Xs = new ArrayList<>();
    static ArrayList<ArrayList> Ys = new ArrayList<>();
    static int numberofinputs, numberofhidden, numberofoutputs;

    public static void read_from_file(String filename) {
        try {
            File myObj = new File(filename);
            Scanner myReader = new Scanner(myObj);
            numberofinputs = myReader.nextInt();
            numberofhidden = myReader.nextInt();
            numberofoutputs = myReader.nextInt();

            int numberofTrainingExamples = myReader.nextInt();
            for (int i = 0; i < numberofTrainingExamples; i++) {
                ArrayList<Double> X = new ArrayList<>();
                ArrayList<Double> Y = new ArrayList<>();
                for (int j = 0; j < numberofinputs; j++) {
                    X.add(myReader.nextDouble());
                }
                for (int j = 0; j < numberofoutputs; j++) {
                    Y.add(myReader.nextDouble());
                }
                Xs.add(X);
                Ys.add(Y);
            }
            System.out.println("number of inputs "+numberofinputs);
            System.out.println("number of hidden "+numberofhidden);
            System.out.println("number of outputs "+numberofoutputs);
            System.out.println("number of training Examples "+numberofTrainingExamples);
            System.out.println("X's and their Y's");
            for(int i=0; i<numberofTrainingExamples; i++)
            {
                for(int j=0; j<numberofinputs; j++)
                {
                    System.out.print("X "+Xs.get(i).get(j)+"     ");
                }
                 for(int j=0; j<numberofoutputs; j++)
                {
                    System.out.print("y "+Ys.get(i).get(j)+"     ");
                }
            }
            
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
        read_from_file("input_3.txt");
    }

}
