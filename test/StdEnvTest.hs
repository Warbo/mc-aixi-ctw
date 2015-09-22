{-# LANGUAGE ExistentialQuantification #-}
import System.Exit
import System.IO
import System.Process
import Test.QuickCheck
import Test.QuickCheck.Monadic

runAixiOn :: String -> IO String
runAixiOn s = do (i, o, e, p) <- spawnAixi
                 hPutStr i (cycle s)
                 out  <- hGetContents o
                 err  <- hGetContents e
                 code <- waitForProcess p
                 hPutStr stderr err
                 case code of
                      ExitSuccess   -> return out
                      ExitFailure n -> error ("AIXI failed with error code " ++
                                              show n)

spawnAixi = runInteractiveProcess "../aixi" [] Nothing Nothing

tests = [let test s = do run $ runAixiOn s
                         assert True
         in T "Can call aixi executable"
              (\s -> not (null s) ==> monadicIO (test s))]

data T = forall a. Testable a => T String a

test (T s t) = putStrLn s >> quickCheck t

main = mapM_ test tests
