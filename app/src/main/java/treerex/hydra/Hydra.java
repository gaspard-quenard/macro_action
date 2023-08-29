/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package treerex.hydra;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import fr.uga.pddl4j.parser.Connector;
import fr.uga.pddl4j.parser.DefaultParsedProblem;
import fr.uga.pddl4j.parser.ErrorManager;
import fr.uga.pddl4j.parser.Expression;
import fr.uga.pddl4j.parser.Location;
import fr.uga.pddl4j.parser.Message;
import fr.uga.pddl4j.parser.ParsedAction;
import fr.uga.pddl4j.parser.ParsedMethod;
import fr.uga.pddl4j.parser.Parser;
import fr.uga.pddl4j.parser.Symbol;
import fr.uga.pddl4j.parser.SymbolType;
import fr.uga.pddl4j.parser.TypedSymbol;
import fr.uga.pddl4j.planners.LogLevel;
import fr.uga.pddl4j.problem.DefaultProblem;
import treerex.hydra.SolverConfig.SolverConfig;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

public class Hydra {

    /**
     * The main method the class. The first argument must be the path to the PDDL
     * domain description and the second
     * argument the path to the PDDL problem description.
     *
     * @param args the command line arguments.
     */
    public String projectDir = "";
    // Configurations used by the solver
    public HashSet<SolverConfig> solverConfigs = new HashSet<SolverConfig>();
    public DefaultProblem problem2;

    private HashMap<String, ParsedAction> parsedActionNameToObj;

    private final Logger LOGGER = LogManager.getLogger(Hydra.class.getName());

    public void run(String[] args) {

        // Set the log level to ALL
        Configurator.setLevel(LOGGER.getName(), Level.ALL);

        String s = System.getProperty("user.dir");
        projectDir = s.replaceAll("\\\\", "/");
        // Checks the number of arguments from the command line
        // If no arguments - we will run the planner on Transport p0.
        // Useful, if we want to quickly debug the program
        if (args.length == 0) {
            System.out.println("No args. Running on ipc2020 TO childsnack p0");
            String dom = projectDir + "/benchmarks/ipc2020/total-order/Childsnack/domain.hddl";
            String p = projectDir + "/benchmarks/ipc2020/total-order/Childsnack/p01.hddl";
            args = new String[] {
                    dom,
                    p };
        }
        // Otherwise, we need 2 arguments - domain and proble mpath
        else if (args.length < 3) {
            System.out.println("Error. Need 3 arguments - sat/smt/csp AND domainPath AND problemPath");
            return;
        }

        // All further argument will be for the configuration of the solver
        for (int i = 3; i < args.length; i++) {
            for (SolverConfig solverConf : SolverConfig.values()) {
                if (args[i].equalsIgnoreCase(solverConf.name())) {
                    solverConfigs.add(solverConf);                
                }
            }
        }

        try {

            // 1. PARSE THE PROBLEM
            // Creates an instance of the PDDL parser
            final Parser parser = new Parser();
            // Disable log
            parser.setLogLevel(LogLevel.OFF);
            // Parses the domain and the problem files.
            final DefaultParsedProblem parsedProblem = parser.parse(args[1], args[2]);
            // Gets the error manager of the parser
            final ErrorManager errorManager = parser.getErrorManager();
            // Checks if the error manager contains errors
            if (!errorManager.isEmpty()) {
                // Prints the errors
                for (Message m : errorManager.getMessages()) {
                    System.out.println(m.toString());
                }
            }

            HashSet<String> primitiveTasks = new HashSet<String>();
            parsedActionNameToObj = new HashMap<String, ParsedAction>();
            for (ParsedAction parsedAction : parsedProblem.getActions()) {
                primitiveTasks.add(parsedAction.getName().getValue());
                parsedActionNameToObj.put(parsedAction.getName().getValue(), parsedAction);
            }

            LOGGER.info("All primitive tasks: " + primitiveTasks.toString() + "\n");


            // Create a dictionary with in key the name of the method and in value an array of double where the first value is the index of the first primitive task and the second value is the index of the last primitive task
            HashMap<ParsedMethod , ArrayList<int[]>> mapMethodsToMacroPossible = new HashMap<ParsedMethod , ArrayList<int[]>>();

            // Ok, now iterate all methods and check if there are consecutives action within the subtask of a method
            for (ParsedMethod method : parsedProblem.getMethods()) {
                LOGGER.info("Analyze method: " + method.getName() + "\n");
                
                // Iterate all subtasks of the method
                int idxFirstPrimitiveTask = -1;
                int idxLastPrimitiveTask = -1;

                mapMethodsToMacroPossible.put(method, new ArrayList<int[]>());

                



                int numSubtasks = method.getSubTasks().getChildren().size();

                for (Integer subtaskIdx = 0; subtaskIdx < numSubtasks; subtaskIdx++) {

                    Expression<String> subtask = method.getSubTasks().getChildren().get(subtaskIdx);
                    
                    // Get the name of the subtask
                    String subtaskName = subtask.getSymbol().getValue();

                    // Check if it is a primitive subtask
                    if (primitiveTasks.contains(subtaskName)) {

                        // Assign the index of the first primitive task (if not already assigned)
                        if (idxFirstPrimitiveTask == -1) {
                            idxFirstPrimitiveTask = subtaskIdx;
                        }

                        // Assign the index of the last primitive task
                        idxLastPrimitiveTask = subtaskIdx;
                    } 
                    // If this is not a primitive task or it is the last subtask of the method
                    if (!primitiveTasks.contains(subtaskName) || subtaskIdx == numSubtasks - 1) {
                        // If the subtask is not primitive, then check if there were consecutive primitive tasks before
                        if (idxFirstPrimitiveTask != -1 && idxLastPrimitiveTask != idxFirstPrimitiveTask) {
                            // There were consecutive primitive tasks, add them to the map
                            LOGGER.info("Found consecutive primitive tasks in method " + method.getName() + " from " + idxFirstPrimitiveTask + " to " + idxLastPrimitiveTask + "\n");
                            mapMethodsToMacroPossible.get(method).add(new int[] {idxFirstPrimitiveTask, idxLastPrimitiveTask});
                            break;
                        }
                    }
                }
                int a = 0;
            }

            // Now iterate our dictionnary 
            for (ParsedMethod method : mapMethodsToMacroPossible.keySet()) {
                for (int[] idxs : mapMethodsToMacroPossible.get(method)) {
                    LOGGER.info("Method " + method.getName() + " has consecutive primitive tasks from " + idxs[0] + " to " + idxs[1] + "\n");

                    // Launch the function that will create the macro actions
                    createMacroActions(method, idxs[0], idxs[1], parsedProblem);
                }
            }


            // Write the new domain (bit of a hack here, the method toPDDLDomainString is private, so to access an equivalent we need to tell that the problem is null)
            parsedProblem.setProblemName(null);
            String newDomainFile = parsedProblem.toString();

            // Write the new domain file
            String newDomainPath = args[1].substring(0, args[1].lastIndexOf("/")) + "/newDomain.hddl";

            File file = new File(newDomainPath);

            if (!file.exists()) {
                file.createNewFile();
            }

            FileWriter writer = new FileWriter(file);
            writer.write(newDomainFile);
            writer.flush();
            writer.close();

            // This exception could happen if the domain or the problem does not exist
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

    // public void createMacroActions(ParsedMethod method, int idxFirstPrimitiveTask, int idxLastPrimitiveTask, DefaultParsedProblem parsedProblem) {

    //     // Create the name of the macro action
    //     String macroActionName = method.getName().getValue() + "_macro";

    //     // Initialize the parameters, preconditions and effects of the macro action
    //     ArrayList<TypedSymbol<String>> macroActionParameters = new ArrayList<TypedSymbol<String>>();
    //     ArrayList<Expression<String>> macroActionPrecondition = new ArrayList<>();
    //     ArrayList<Expression<String>> macroActionEffect = new ArrayList<>();

    //     ArrayList<Expression<String>> worldState = new ArrayList<>();

    //     for (Integer subtaskToAdd = idxFirstPrimitiveTask; subtaskToAdd <= idxLastPrimitiveTask; subtaskToAdd++) {

    //         // Get the subtask
    //         Expression<String> subtask = method.getSubTasks().getChildren().get(subtaskToAdd);

    //         LOGGER.info("===========\n===========\nAdd subtask " + subtask + " to macro action " + macroActionName + "\n");

    //         // Get a copy of the parsed action associated with this subtask
    //         ParsedAction parsedAction = new ParsedAction(parsedActionNameToObj.get(subtask.getSymbol().getValue()));

    //         // First, change all its parameters (in parameter field, precondition and effect) with the parameters of the macro action method.
    //         // changeParametersActionWithParameterMethod(parsedAction, method.getName().getValue(), subtask.getArguments());

    //         // Then iterate over all its preconditions
    //         for (int i = 0; i < parsedAction.getPreconditions().getChildren().size(); i ++) {

    //             Expression<String> precondition = parsedAction.getPreconditions().getChildren().get(i);

    //             LOGGER.debug("===============\n");
    //             LOGGER.debug("World state: " + worldState + "\n");
    //             LOGGER.debug("Precondition macro action: " + macroActionPrecondition + "\n");
    //             LOGGER.debug(macroActionName + " - Precondition: " + precondition + "\n");

    //             // 4 cases here:
    //             // - The precondition of the macro action contains the precondition and the world state contains the precondition => NOTHING TO DO
    //             // - The precondition of the macro action does not contain the precondition and the world state contains the precondition => NOTHING TO DO: the precondition is always verified
    //             // RESUME FIRST TWO CASES: If the world state contains the precondition, then it is always verified, so we do not need to add it to the macro action precondition

    //             // - The precondition of the macro action contains the precondition and the world state does not contain the precondition (or contains the opposite of the precondition) => SHOULD BE IMPOSSIBLE: ERROR
    //             // - The precondition of the macro action does not contain the precondition and the world state does not contain the precondition => ADD THE PRECONDITION TO THE MACRO ACTION PRECONDITION AND TO THE WORLD STATE

    //             if (worldState.contains(precondition)) {
    //                 // We can skip this precondition (will be always verified)
    //                 LOGGER.debug("Skip this precondition (will be always verified)\n");
    //                 continue;
    //             }
    //             else if (containsOppositeOf(precondition, worldState)) {
    //                 // Error here: the precondition of the macro action contains the precondition and the world state contains the opposite of the precondition: this precondition will never be verified
    //                 LOGGER.error("Error: the precondition of the macro action " + macroActionName + " contains the precondition " + precondition + " and the world state contains the opposite of the precondition: this precondition will never be verified\n");
    //                 System.exit(1);
    //             }
    //             else {
    //                 // Add the precondition into the macro action precondition (if not exists)
    //                 if (!macroActionPrecondition.contains(precondition)) {
    //                     macroActionPrecondition.add(precondition);
    //                 }

    //                 // Add it to the world state
    //                 worldState.add(precondition);
    //             }
    //         }

    //         // Now, iterate over all effects to update the world state
    //         for (Expression<String> effect : parsedAction.getEffects().getChildren()) {
                    
    //                 LOGGER.debug("===============\n");
    //                 LOGGER.debug("World state: " + worldState + "\n");
    //                 LOGGER.debug(macroActionName + " - Effect: " + effect + "\n");
    
    
    //                 if (worldState.contains(effect)) {
    //                     // We can skip this effect (will be always verified)
    //                     continue;
    //                 }
    //                 else if (containsOppositeOf(effect, worldState)) {
    //                     // Remove this effect from the world state
    //                     removeOppositeOf(effect, worldState);
    //                 }
    //                 // Add the effect into the world state
    //                 worldState.add(effect);
    //         }

    //         int a = 0;
    //     }

    //     int a = 0;


    // }

    /**
     * Create the macro actions for a given method. We use the following technique to create the macro actions:
     * To create a macro action A with a1 and a2:
     * Pre(A) = pre(a1) U (pre(a2) \ add(a1))
     * Del(A) = (del(a1) U add(a2)) \ del(a1)
     * Add(A) = (add(a1) U del(a2)) \ add(a1)
     * If the macro action contains more thant 2 actions, the macro action can be construct by iteratively using this algorithm
     * @param method
     * @param idxFirstPrimitiveTask
     * @param idxLastPrimitiveTask
     * @param parsedProblem
     */
    public void createMacroActions(ParsedMethod method, int idxFirstPrimitiveTask, int idxLastPrimitiveTask, DefaultParsedProblem parsedProblem) {

        // Create the name of the macro action
        String macroActionName = "Macro-" +  method.getName().getValue() + "__" + idxFirstPrimitiveTask + "-" + idxLastPrimitiveTask + "_";

        // Initialize the parameters, preconditions and effects of the macro action
        ArrayList<TypedSymbol<String>> macroActionParameters = new ArrayList<TypedSymbol<String>>();
        Expression<String> macroActionPreconditions = new Expression<String>();
        Expression<String> macroActionAddEffects = new Expression<String>();
        Expression<String> macroActionDelEffects = new Expression<String>();

        ArrayList<TypedSymbol<String>> newMacroActionParameters = new ArrayList<TypedSymbol<String>>();
        Expression<String> newMacroActionPreconditions = new Expression<String>();
        Expression<String> newMacroActionAddEffects = new Expression<String>();
        Expression<String> newMacroActionDelEffects = new Expression<String>();

        // First change all the parameters of all the actions which will be in the macro action with the parameter of the method
        Queue<ParsedAction> parsedActionsInMacroAction = new LinkedList<ParsedAction>();
        for (Integer subtaskToAdd = idxFirstPrimitiveTask; subtaskToAdd <= idxLastPrimitiveTask; subtaskToAdd++) {

            macroActionName += "__" + method.getSubTasks().getChildren().get(subtaskToAdd).getSymbol().getValue();

            // Get the subtask
            Expression<String> subtask = method.getSubTasks().getChildren().get(subtaskToAdd);

            LOGGER.info("===========\n===========\nAdd subtask " + subtask + " to macro action " + macroActionName + "\n");

            // Get a copy of the parsed action associated with this subtask
            ParsedAction parsedAction = new ParsedAction(parsedActionNameToObj.get(subtask.getSymbol().getValue()));

            // First, change all its parameters (in parameter field, precondition and effect) with the parameters of the macro action method.
            changeParametersActionWithParameterMethod(parsedAction, method.getName().getValue(), subtask.getArguments());

            // Add the parsed action in the list
            // parsedActionsInMacroAction.add(parsedAction);

            // Add the parameters, preconditions and effects to the macro action
            newMacroActionParameters = addParamsToMacro(macroActionParameters, parsedAction.getParameters());
            newMacroActionPreconditions = addPreToMacro(macroActionPreconditions, macroActionAddEffects, parsedAction.getPreconditions());
            newMacroActionDelEffects = addDelToMacro(macroActionDelEffects, parsedAction.getEffects());
            newMacroActionAddEffects = addAddToMacro(macroActionAddEffects, parsedAction.getEffects());

            // Update the macro action parameters, preconditions, add effects and del effects
            macroActionParameters = newMacroActionParameters;
            macroActionPreconditions = newMacroActionPreconditions;
            macroActionAddEffects = newMacroActionAddEffects;
            macroActionDelEffects = newMacroActionDelEffects;

            LOGGER.debug(" - Params: " + newMacroActionParameters + "\n\n");
            LOGGER.debug(" - Pre: " + macroActionPreconditions + "\n\n");
            LOGGER.debug(" - Add: " + macroActionAddEffects + "\n\n");
            LOGGER.debug(" - Del: " + macroActionDelEffects + "\n\n");
        }

        // It works, but there are still artefacts (effects that are not needed). For example, with the childsnack problem, we have:
        /* *macro action Macro:m1_serve__[0,4]___make_sandwich__put_on_tray__move_tray__serve_sandwich__move_tray
            - Pre: (and (at_kitchen_bread m1_serve__?b)
            (at_kitchen_content m1_serve__?cont)
            (notexist m1_serve__?s)
            (at m1_serve__?t kitchen)
            (not_allergic_gluten m1_serve__?c)
            (waiting m1_serve__?c m1_serve__?p2))

            - Add: (and (served m1_serve__?c)
            (at m1_serve__?t kitchen))

            - Del: (and (not (at_kitchen_bread m1_serve__?b))
            (not (at_kitchen_content m1_serve__?cont))
            (not (notexist m1_serve__?s))
            (not (at_kitchen_sandwich m1_serve__?s))
            (not (ontray m1_serve__?s m1_serve__?t))
            (not (at m1_serve__?t m1_serve__?p2)))
         * 
         * 
         * We see that in the add effect, (at m1_serve__?t kitchen) is not usefull (already in the precondition, can be discarded)
         * More problematic, in the del effect, we have two artefacts:
         * (not (ontray m1_serve__?s m1_serve__?t)) => not usefull, was already negative in the beginning, but how to prove that ?
         * (not (at m1_serve__?t m1_serve__?p2))) => we have move the tray back to its original position, so this effect is not needed. But how to prove that ?
         */


        // Use a general function to remove the effects that are already in the precondition (function with general name to indicate that we remove all the element of the first list that are in the second list)
        // macroActionAddEffects = filterBy(macroActionAddEffects, macroActionPreconditions);
        

        // Fusion the add and del effects
        Expression<String> macroActionEffects = new Expression<String>(macroActionAddEffects);
        for (Expression<String> macroActionDelEff : macroActionDelEffects.getChildren()) {
            macroActionEffects.addChild(macroActionDelEff);
        }

        Symbol<String> symbolMacroActionName = new Symbol<String>(SymbolType.ACTION, macroActionName);


        // Create the macro action
        ParsedAction macroAction = new ParsedAction(symbolMacroActionName, macroActionParameters, macroActionPreconditions, macroActionEffects);
        
        // Add the macro action to the parsed problem
        parsedProblem.getActions().add(macroAction);

        // Finally, update the method to replace the subtasks by the macro action
        Expression<String> newSubtasks = new Expression<String>();
        for (int i = 0; i < idxFirstPrimitiveTask; i++) {
            newSubtasks.addChild(method.getSubTasks().getChildren().get(i));
        }

        Expression<String> expressionMacro = new Expression<String>();
        expressionMacro.setSymbol(symbolMacroActionName);
        List<Symbol<String>> macroActionParametersForMethod = new ArrayList<Symbol<String>>();
        for (TypedSymbol<String> param : macroAction.getParameters()) {
            macroActionParametersForMethod.add(param);
        }
        expressionMacro.setArguments(macroActionParametersForMethod);
        expressionMacro.setConnector(Connector.TASK);
        // expressionMacro.setLocation(new Location(25, 28, 56, 28));
        expressionMacro.setTaskID(new Symbol<String>(SymbolType.TASK_ID, "t" + (idxFirstPrimitiveTask + 1)));

        newSubtasks.addChild(expressionMacro);

        // newSubtasks.addChild(symbolMacroActionName);
        for (int i = idxLastPrimitiveTask+1; i < method.getSubTasks().getChildren().size(); i++) {
            Expression<String> subtask = method.getSubTasks().getChildren().get(i);
            // We have to change the taks id since we have removed some subtasks
            subtask.setTaskID(new Symbol<String>(SymbolType.TASK_ID, "t" + (i - (idxLastPrimitiveTask - idxFirstPrimitiveTask))));
            newSubtasks.addChild(subtask);
        }

        method.setSubTasks(newSubtasks);
    }


    /**
     * Change the parameters of the action with the parameters of the method
     * @param parsedAction The action to change
     * @param nameMethod The name of the method
     * @param methodParametersForAction The parameters with which the method call the action (in the same order as the parameters of the action)
     */
    public void changeParametersActionWithParameterMethod(ParsedAction parsedAction, String nameMethod, List<Symbol<String>> methodParametersForAction) {

        // Verify that the number of parameters in the method is the same as the number of parameters in the action
        if (parsedAction.getParameters().size() != methodParametersForAction.size()) {
            LOGGER.error("The number of parameters in the method " + nameMethod + " is different from the number of parameters in the action " + parsedAction.getName().getValue() + "\n");
            System.exit(1);
        }

        // First, make a dictionary which maps the name of the parameter in the action to the name of the parameter in the method
        HashMap<String, String> mapParameterNameActionToParameterNameMethod = new HashMap<String, String>();
        for (int paramIdx = 0; paramIdx < parsedAction.getParameters().size(); paramIdx++) {
            String nameParameterInMethod = methodParametersForAction.get(paramIdx).getValue();
            if (nameParameterInMethod.contains("?")) {
                // This is not a static parameter, use a special name to indicate that it is a parameter from the method
                // nameParameterInMethod = nameMethod + "__" + nameParameterInMethod;
            }

            mapParameterNameActionToParameterNameMethod.put(parsedAction.getParameters().get(paramIdx).getValue(), nameParameterInMethod);
        }

        // Then change the name of the parameters in the action
        for (int paramIdx = 0; paramIdx < parsedAction.getParameters().size(); paramIdx++) {
            // Get the name of the parameter in the action
            String parameterNameInAction = parsedAction.getParameters().get(paramIdx).getValue();

            // Get the name of the parameter in the method
            String parameterNameInMethod = mapParameterNameActionToParameterNameMethod.get(parameterNameInAction);

            // Change the name of the parameter in the action
            parsedAction.getParameters().get(paramIdx).setValue(parameterNameInMethod);
        }

        // Then change the name of the parameters in the precondition
        for (Expression<String> precondition : parsedAction.getPreconditions().getChildren()) {

            if (precondition.getConnector().equals(Connector.NOT)) {
                precondition = precondition.getChildren().get(0);
            }
            
            // Iterate over all arguments of the precondition
            for (Symbol<String> paramInPrecondition : precondition.getArguments()) {

                // The parameter of the precondition can be static and thus not be in the parameter of the action
                if (mapParameterNameActionToParameterNameMethod.containsKey(paramInPrecondition.getValue())) {
                    // Change the name of the parameter in the precondition
                    paramInPrecondition.setValue(mapParameterNameActionToParameterNameMethod.get(paramInPrecondition.getValue()));
                }
            }
        }

        // Then change the name of the parameters in the effect
        for (Expression<String> effect : parsedAction.getEffects().getChildren()) {

            if (effect.getConnector().equals(Connector.NOT)) {
                effect = effect.getChildren().get(0);
            }
            
            // Iterate over all arguments of the effect
            for (Symbol<String> paramInEffect : effect.getArguments()) {
                // Change the name of the parameter in the effect
                paramInEffect.setValue(mapParameterNameActionToParameterNameMethod.get(paramInEffect.getValue()));
            }
        }
    }


    public ArrayList<TypedSymbol<String>> addParamsToMacro(ArrayList<TypedSymbol<String>> macroActionParameters, List<TypedSymbol<String>> actionToAddParameters) {

        ArrayList<TypedSymbol<String>> newMacroActionParameters = new ArrayList<TypedSymbol<String>>(macroActionParameters);

        // Iterate over all parameters of the action to add and add it to the new macro action parameters if this parameter is not already in the macro action parameters
        for (TypedSymbol<String> actionToAddParameter : actionToAddParameters) {

            // If the parameter is a static parameter, skip it
            if (!actionToAddParameter.getValue().contains("?")) {
                continue;
            }

            // If this parameter is already in the macro action parameters, skip it
            boolean isInMacroActionParameters = false;
            for (TypedSymbol<String> macroActionParameter : macroActionParameters) {
                if (macroActionParameter.getValue().equals(actionToAddParameter.getValue())) {
                    isInMacroActionParameters = true;
                    break;
                }
            }
            if (isInMacroActionParameters) {
                continue;
            }

            // Add the parameter to the new macro action parameters
            newMacroActionParameters.add(actionToAddParameter);
        }

        return newMacroActionParameters;
    }

    /**
     * Add preconditions to the macro action with the formula:
     * pre(macroAction, actionToAdd) = pre(macroAction) U (pre(actionToAdd) \ add(macroAction))
     * @param macroActionPreconditions Precondition of the macro action
     * @param macroActionAddEffects Add effects of the macro action
     * @param actionToAddPreconditions Precondition of the action to add
     */
    public Expression<String> addPreToMacro(Expression<String> macroActionPreconditions, Expression<String> macroActionAddEffects, Expression<String> actionToAddPreconditions) {

        Expression<String> newMacroActionPreconditions = new Expression<String>(macroActionPreconditions);
        // Iterate over all precondition of the action to add and add it to the new macro action precondition if this precondition is not in the add effect of the macro action
        for (Expression<String> actionToAddPrecondition : actionToAddPreconditions.getChildren()) {

            // If this action is already in the precondition of the macro action, skip it
            boolean isInPreconditionOfMacroAction = false;
            for (Expression<String> macroActionPrecondition : macroActionPreconditions.getChildren()) {
                if (macroActionPrecondition.equals(actionToAddPrecondition)) {
                    isInPreconditionOfMacroAction = true;
                    break;
                }
            }

            if (isInPreconditionOfMacroAction) {
                continue;
            }

            // If this precondition is in the add effect of the macro action, skip it
            boolean isInAddEffectOfMacroAction = false;
            for (Expression<String> macroActionAddEffect : macroActionAddEffects.getChildren()) {
                if (macroActionAddEffect.equals(actionToAddPrecondition)) {
                    isInAddEffectOfMacroAction = true;
                    break;
                }
            }

            if (!isInAddEffectOfMacroAction) {
                newMacroActionPreconditions.addChild(actionToAddPrecondition);
            }
        }

        return newMacroActionPreconditions;
    }


    /**
     * Add del effects to the macro action with the formula:
     * del(macroAction, actionToAdd) = del(actionToAdd) U (del(macroAction) \ add(actionToAdd))
     * @param macroActionDelEffects Del effects of the macro action
     * @param actionToAddAddAndDelEffects Del and Add effects of the action to add
     */
    public Expression<String> addDelToMacro(Expression<String> macroActionDelEffects, Expression<String> actionToAddAddAndDelEffects) {

        Expression<String> newMacroActionDelEffects = new Expression<String>(macroActionDelEffects);

        // Add all the del effects of the action to add
        for (Expression<String> actionToAddDelEffect : actionToAddAddAndDelEffects.getChildren()) {

            // If this is not a del effect, continue
            if (!actionToAddDelEffect.getConnector().equals(Connector.NOT)) {
                continue;
            }

            // Add it if it is not already in the del effects of the macro action
            boolean isInDelEffectsOfMacroAction = false;
            for (Expression<String> macroActionDelEffect : macroActionDelEffects.getChildren()) {
                if (macroActionDelEffect.equals(actionToAddDelEffect)) {
                    isInDelEffectsOfMacroAction = true;
                    break;
                }
            }

            if (!isInDelEffectsOfMacroAction) {
                newMacroActionDelEffects.addChild(actionToAddDelEffect);
            }
        }

        // Now, remove all the del effects of the macro action if they are in the add effects of the action to add
        for (Expression<String> actionToAddAddEffect : actionToAddAddAndDelEffects.getChildren()) {
            
            // If this is not an add effect, continue
            if (actionToAddAddEffect.getConnector().equals(Connector.NOT)) {
                continue;
            }

            // Remove it if it is in the del effects of the macro action
            for (Expression<String> macroActionDelEffect : macroActionDelEffects.getChildren()) {
                if (isOppositeOf(macroActionDelEffect, actionToAddAddEffect)) {
                    newMacroActionDelEffects.getChildren().remove(macroActionDelEffect);
                    break;
                }
            }
        }

        return newMacroActionDelEffects;
    }


    /**
     * Add add effects to the macro action with the formula:
     * add(macroAction, actionToAdd) = add(actionToAdd) U (add(macroAction) \ del(actionToAdd))
     * @param macroActionAddEffects Add effects of the macro action
     * @param actionToAddAddAndDelEffects Del and Add effects of the action to add
     */
    public Expression<String> addAddToMacro(Expression<String> macroActionAddEffects, Expression<String> actionToAddAddAndDelEffects) {

        Expression<String> newMacroActionAddEffects = new Expression<String>(macroActionAddEffects);

        // Add all the del effects of the action to add
        for (Expression<String> actionToAddAddEffect : actionToAddAddAndDelEffects.getChildren()) {

            // If this is not a add effect, continue
            if (actionToAddAddEffect.getConnector().equals(Connector.NOT)) {
                continue;
            }

            // Add it if it is not already in the add effects of the macro action
            boolean isInAddEffectsOfMacroAction = false;
            for (Expression<String> macroActionDelEffect : macroActionAddEffects.getChildren()) {
                if (macroActionDelEffect.equals(actionToAddAddEffect)) {
                    isInAddEffectsOfMacroAction = true;
                    break;
                }
            }

            if (!isInAddEffectsOfMacroAction) {
                newMacroActionAddEffects.addChild(actionToAddAddEffect);
            }
        }

        // Now, remove all the add effects of the macro action if they are in the del effects of the action to add
        for (Expression<String> actionToAddDelEffect : actionToAddAddAndDelEffects.getChildren()) {
            
            // If this is not an del effect, continue
            if (!actionToAddDelEffect.getConnector().equals(Connector.NOT)) {
                continue;
            }

            // Remove it if it is in the del effects of the macro action
            for (Expression<String> macroActionAddEffect : macroActionAddEffects.getChildren()) {
                if (isOppositeOf(macroActionAddEffect, actionToAddDelEffect)) {
                    newMacroActionAddEffects.getChildren().remove(macroActionAddEffect);
                    break;
                }
            }
        }

        return newMacroActionAddEffects;
    }

    /**
     * Check if the world state contains the opposite of the expression
     * @param expression The expression
     * @param worldState The world state
     * @return True if the world state contains the opposite of the expression, false otherwise
     */
    public boolean containsOppositeOf(Expression<String> expression, ArrayList<Expression<String>> worldState) {

        // If the expression is positive(negative), search for the negative(positive) expression 
        Expression<String> oppositeExpression = new Expression<String>(expression);
        
        if (expression.getConnector().equals(Connector.NOT)) {
            oppositeExpression = expression.getChildren().get(0);
        } else {
            oppositeExpression.setConnector(Connector.NOT);
        }

        // Iterate over all expressions in the world state
        for (Expression<String> expressionInWorldState : worldState) {

            // If the expression in the world state is the opposite of the expression, return true
            if (expressionInWorldState.equals(oppositeExpression)) {
                return true;
            }
        }

        // If we did not find the opposite of the expression in the world state, return false
        return false;
    }

    /**
     * Remove the opposite of the expression from the world state
     * @param expression The expression
     * @param worldState The world state
     */
    public void removeOppositeOf(Expression<String> expression, ArrayList<Expression<String>> worldState) {

        // If the expression is positive(negative), search for the negative(positive) expression 
        Expression<String> oppositeExpression = new Expression<String>(expression);
        
        if (expression.getConnector().equals(Connector.NOT)) {
            oppositeExpression = expression.getChildren().get(0);
        } else {
            oppositeExpression.setConnector(Connector.NOT);
        }

        // Iterate over all expressions in the world state
        for (Expression<String> expressionInWorldState : worldState) {

            // If the expression in the world state is the opposite of the expression, remove it
            if (expressionInWorldState.equals(oppositeExpression)) {
                worldState.remove(expressionInWorldState);
                return;
            }
        }
    }

    public boolean isOppositeOf(Expression<String> expression1, Expression<String> expression2) {

        // If the expressions are positive(negative), return true if they are the same
        if (expression1.getConnector().equals(Connector.NOT)) {
            return expression1.getChildren().get(0).equals(expression2);
        }

        if (expression2.getConnector().equals(Connector.NOT)) {
            return expression2.getChildren().get(0).equals(expression1);
        }

        return false;
    }

    // public Expression<String> filterBy(Expression<String> expressionToFilter, Expression<String> filter) {

    //     Expression<String> newExpression = new Expression<String>(expression);

    //     for (Expression<String> element : expression.getChildren()) {
            
    //         // If this element is in the filter, remove it
    //         if (el)
    //     }

    //     return newExpression;
    // }
}