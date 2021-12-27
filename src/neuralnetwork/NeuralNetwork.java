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
import java.util.Arrays;
import java.util.Collections;

/**
 *
 * @author Magooda
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    static public class record {
        
        ArrayList<Double> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();
        
        public record(ArrayList x, ArrayList y) {
            X = x;
            Y = y;
        }
        
    }
    static ArrayList<ArrayList> Xs = new ArrayList<>();
    static ArrayList<ArrayList> Ys = new ArrayList<>();
    static int numberofinputs, numberofhidden, numberofoutputs, numberofTrainingExamples;
    
    public static void read_from_file(String filename) {
        try {
            File myObj = new File(filename);
            Scanner myReader = new Scanner(myObj);
            numberofinputs = myReader.nextInt();
            numberofhidden = myReader.nextInt();
            numberofoutputs = myReader.nextInt();
            
            numberofTrainingExamples = myReader.nextInt();
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
            /*     System.out.println("number of inputs "+numberofinputs);
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
             */
            
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        
    }
    
    public static void Normalization() {
        ArrayList<ArrayList> newxs = new ArrayList<>();
        ArrayList<Double> means = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        ArrayList<Double> Sd = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        Collections.fill(means, 0.0);
        Collections.fill(Sd, 0.0);
        Double value = 0.0, sumTillNow = 0.0, sdTillNow = 0.0;
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {
                value = (Double) Xs.get(i).get(j);
                sumTillNow = means.get(j);
                means.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    means.add(j, ((value + sumTillNow) / numberofTrainingExamples));
                } else {
                    means.add(j, (value + sumTillNow));
                }
                
            }
            
        }
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {
                
                Double mean = means.get(j);
                value = (Double) Xs.get(i).get(j);
                sdTillNow = Sd.get(j);
                
                Sd.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    Double Val = (value - mean) * (value - mean) + sdTillNow;
                    Val = Val / numberofTrainingExamples;
                    Sd.add(j, Math.sqrt(Val));
                } else {
                    
                    Sd.add(j, (Double) ((value - mean) * (value - mean)) + sdTillNow);
                }
                
            }
            
        }
       /* System.out.println("mean");
        for(int i=0; i<means.size(); i++)
        {
            System.out.println(means.get(i));
        }
        System.out.println("Sd");
        for(int i=0; i<Sd.size(); i++)
        {
            System.out.println(Sd.get(i));
        }
        */
        for (int i = 0; i < numberofTrainingExamples; i++) {
            ArrayList<Double> newx = new ArrayList<>();
            for (int j = 0; j < numberofinputs; j++) {
                Double new_x = (((Double) Xs.get(i).get(j) - means.get(j)) / Sd.get(j));
                newx.add(new_x);
                
            }
            newxs.add(newx);
        }
        
        Xs = newxs;
        
    }
    
    public static void main(String[] args) {
        read_from_file("input_3.txt");
        Normalization();
        
       
        
    }
    
}
